"""vLLM predictor with the same prompt, labels, readout and temperatures as ``LMPredictor``.

    from ajev.lm.vllm_predictor import VLLMPredictor
    probs = VLLMPredictor("google/gemma-4-12B-it", adapter="andyzhang232/ajev-gemma4-12b-lora5").predict(decisions)

Prompt token ids are built locally and passed to vLLM. One token is sampled with ``allowed_token_ids`` set to
the labels; their log-probabilities are read back, the two forms of each label are combined, and the result is
divided by the type temperature. The LoRA is loaded directly (do not serve a saved merged Gemma 4 checkpoint).
Prompts over ``max_model_len`` raise ``ContextTooLong``.
"""

from __future__ import annotations

import math
import uuid

from ajev.lm.labels import option_labels
from ajev.lm.predictor import load_tokenizer, prompt_ids, read_lm_config, resolve_adapter
from ajev.lm.prompt import MAX_OPTIONS
from ajev.schema import Decision

NO_TRUNCATION = 10 ** 9


class ContextTooLong(ValueError):
    pass


def engine_kwargs(model_id: str, adapter: str | None, max_model_len: int, gpu_memory_utilization: float) -> dict:
    """Engine arguments; max_logprobs covers 255 labels x 2 forms."""
    kw = dict(model=model_id, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
              enable_prefix_caching=True, max_logprobs=2 * MAX_OPTIONS + 2, logprobs_mode="processed_logprobs")
    if adapter:
        kw.update(enable_lora=True, max_lora_rank=64, max_loras=1)
    return kw


class LabelReader:
    """Prompt ids, sampling params and probability readout, shared by both engines."""

    def __init__(self, model_id: str, adapter: str | None, max_model_len: int,
                 temperatures: dict[str, float] | None = None) -> None:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        self.SamplingParams = SamplingParams
        self.tok = load_tokenizer(model_id)
        self.max_model_len = max_model_len
        cfg = read_lm_config(adapter)
        self.temperatures = temperatures if temperatures is not None else cfg.get("temperatures", {})
        self.lora = LoRARequest("ajev", 1, adapter) if adapter else None
        _, ids, valid = option_labels(self.tok)
        self.forms = [[ids[i][j] for j in range(2) if valid[i][j]] for i in range(len(ids))]

    def prompt(self, d: Decision) -> list[int]:
        ids = prompt_ids(self.tok, d, NO_TRUNCATION)
        if len(ids) + 1 > self.max_model_len:
            raise ContextTooLong(f"prompt of {len(ids)} tokens exceeds the maximum context length {self.max_model_len}")
        return ids

    def params(self, d: Decision):
        if len(d.options) > len(self.forms):
            raise ValueError(f"{len(d.options)} options per choice > {len(self.forms)} labels")
        allowed = sorted({t for f in self.forms[: len(d.options)] for t in f})
        return self.SamplingParams(max_tokens=1, temperature=0, logprobs=len(allowed), allowed_token_ids=allowed)

    def probs(self, d: Decision, output) -> list[float]:
        lp = output.outputs[0].logprobs[0]  # {token_id: Logprob} over the allowed labels
        scores = []
        for f in self.forms[: len(d.options)]:
            vals = [lp[t].logprob for t in f if t in lp] or [-1e4]
            m = max(vals)
            scores.append(m + math.log(sum(math.exp(v - m) for v in vals)))
        t = self.temperatures.get(d.type, 1.0)
        m = max(scores)
        e = [math.exp((s - m) / t) for s in scores]
        z = sum(e)
        return [x / z for x in e]


class VLLMPredictor:
    """Offline batch inference with the LMPredictor interface."""

    def __init__(self, model_id: str, adapter: str | None = None, max_model_len: int = 131072,
                 gpu_memory_utilization: float = 0.88, temperatures: dict[str, float] | None = None) -> None:
        from vllm import LLM

        adapter = resolve_adapter(adapter)
        self.reader = LabelReader(model_id, adapter, max_model_len, temperatures)
        self.tok, self.temperatures = self.reader.tok, self.reader.temperatures
        self.llm = LLM(**engine_kwargs(model_id, adapter, max_model_len, gpu_memory_utilization))

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        from vllm.inputs import TokensPrompt

        r = self.reader
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=r.prompt(d)) for d in decisions],
                                 [r.params(d) for d in decisions], lora_request=r.lora, use_tqdm=False)
        return [r.probs(d, o) for d, o in zip(decisions, outs)]


class AsyncVLLMPredictor:
    """Async inference for the HTTP server."""

    def __init__(self, model_id: str, adapter: str | None = None, max_model_len: int = 131072,
                 gpu_memory_utilization: float = 0.88) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        adapter = resolve_adapter(adapter)
        self.reader = LabelReader(model_id, adapter, max_model_len)
        self.tok, self.temperatures = self.reader.tok, self.reader.temperatures
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs(model_id, adapter, max_model_len,
                                                                                 gpu_memory_utilization)))

    async def _one(self, d: Decision, ids: list[int]) -> list[float]:
        from vllm.inputs import TokensPrompt

        final = None
        async for out in self.engine.generate(TokensPrompt(prompt_token_ids=ids), self.reader.params(d),
                                              request_id=uuid.uuid4().hex, lora_request=self.reader.lora):
            final = out
        return self.reader.probs(d, final)

    async def predict(self, decisions: list[Decision]) -> tuple[list[list[float]], int]:
        """(probabilities per decision, total input tokens); questions are submitted together."""
        import asyncio

        ids = [self.reader.prompt(d) for d in decisions]
        probs = await asyncio.gather(*(self._one(d, i) for d, i in zip(decisions, ids)))
        return list(probs), sum(len(i) for i in ids)
