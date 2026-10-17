"""Decision Index engine (github.com/apolinario/decision-index), in process.

    python -m decision_index run --engine ajev.jdi_engine:AJevEngine \\
        --option adapter=andyzhang232/ajev-gemma4-12b-lora5 --out runs/ajev

Same prompt and readout as everyday use. States are never truncated: requests over the context limit,
or that do not fit in GPU memory, are reported as unsupported. All options are scored.
``--option thinking=off|on|auto`` (default off) selects the thinking mode (see ajev/lm/thinking.py).
``--option prompt_version=v1|v2`` overrides the prompt format (default: the adapter's, else v1; see ajev/lm/prompt.py).
"""

from __future__ import annotations

from decision_index.engines.base import Engine, Unsupported

from ajev.jev_request import detect_lang, plain_questions
from ajev.schema import NOUL_TRUE, decisions_from_jev

NO_TRUNCATION = 10 ** 9


class AJevEngine(Engine):
    name = "ajev"
    latency = "Device-synchronized in-process request wall time including prompt construction; all questions of a request scored in token-budgeted batches; excludes model loading."

    def __init__(self, model="google/gemma-4-12B-it", adapter=None, batch_tokens=24000, prefix_cache=True,
                 thinking="off", prompt_version=None, **options):
        super().__init__(**options)
        import torch

        from ajev.lm.predictor import LMPredictor, prompt_ids
        from ajev.lm.thinking import parse_mode

        self.torch, self.prompt_ids = torch, prompt_ids
        prefix_cache = prefix_cache not in (False, "false", "False", "0", 0)
        self.thinking = parse_mode(thinking)
        self.p = LMPredictor(model, adapter=adapter or None, batch_tokens=int(batch_tokens),
                             max_state_tokens=NO_TRUNCATION, prefix_cache=prefix_cache,
                             prompt_version=prompt_version or None)
        self.limit = self.p.context_limit  # 和 LMPredictor 的思考预算用同一个上限
        if self.thinking != "off" and self.p.close_id is None:
            raise ValueError("thinking is not supported by this base model")
        self.model_id = f"ajev:{adapter or model}"
        self.provenance = {
            "kind": "ajev", "base_model": model, "adapter": adapter, "context_limit_tokens": self.limit,
            "temperatures": self.p.temperatures, "thinking": self.thinking, "prompt_version": self.p.prompt_version,
            "policy": "Each question is one prompt (state + question + labelled options); the next-token logits of the "
                      "option labels (A-Z, then single-token two-letter codes up to 255) are softmaxed with a per-type "
                      "calibration temperature. Questions of one request share the token-identical prompt prefix "
                      f"(instructions + state) through a KV cache computed once (prefix_cache={prefix_cache}). "
                      "No truncation: prompts over the context limit, or that do not fit in "
                      "GPU memory, are refused as unsupported."
                      + ("" if self.thinking == "off" else
                         f" Thinking={self.thinking}: a greedy thought of at most {self.p.thinking.budget} tokens "
                         "(closed by force at the budget) precedes the label readout"
                         + (f" for questions whose top probability is below {self.p.thinking.auto_threshold}"
                            if self.thinking == "auto" else "") + "."),
        }

    def runtime(self):
        t = self.torch
        info = {"torch": t.__version__, "device": str(self.p.device)}
        if t.cuda.is_available():
            info.update(cuda=t.version.cuda, gpu=t.cuda.get_device_name())
        return info

    def synchronize(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def __call__(self, state, questions):
        for k, q in questions.items():
            if q["type"] not in ("choice", "noul"):
                raise Unsupported(f"question {k}: unsupported type {q['type']}")
        ds = decisions_from_jev(state, plain_questions(questions), lang=detect_lang(state, questions))
        lengths = [len(self.prompt_ids(self.p.tok, d, NO_TRUNCATION, version=self.p.prompt_version)) for d in ds]
        if max(lengths) > self.limit:
            raise Unsupported(f"prompt of {max(lengths)} tokens exceeds the {self.limit}-token context")
        try:
            probs = self.p.predict(ds, thinking=self.thinking)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            raise Unsupported(f"out of GPU memory at {max(lengths)} prompt tokens")
        answers = {}
        for d, ps in zip(ds, probs):
            pmap = dict(zip(d.option_names, ps))
            qid = d.meta["question_id"]
            if d.type == "noul":
                answers[qid] = {"type": "noul", "noul": min(1.0, max(0.0, pmap[NOUL_TRUE]))}
            else:
                best = max(pmap, key=pmap.get)
                answers[qid] = {"type": "choice", "choice": best, "probabilities": pmap}
        usage = {"input_tokens": sum(lengths)}
        if self.thinking != "off":
            usage["output_tokens"] = sum(t.tokens for t in self.p.last_thoughts)
        return {"model": self.model_id, "answers": answers, "usage": usage}, None
