"""vLLM predictor with the same prompt, labels, readout and temperatures as ``LMPredictor``.

    from ajev.lm.vllm_predictor import VLLMPredictor
    probs = VLLMPredictor("google/gemma-4-12B-it", adapter="andyzhang232/ajev-gemma4-12b-lora5").predict(decisions)

Prompt token ids are built locally and passed to vLLM. One token is sampled with ``allowed_token_ids`` set to
the labels; their log-probabilities are read back, the two forms of each label are combined, and the result is
divided by the type temperature. The LoRA is loaded directly (do not serve a saved merged Gemma 4 checkpoint).
Prompts over ``max_model_len`` raise ``ContextTooLong``.

Thinking (``thinking="on"`` / ``"auto"``, see ajev/lm/thinking.py): the thought is generated greedily up to the
budget and stops at the close token; the labels are then read on prompt + thought, which reuses the thought's
KV cache through prefix caching. A question with no room left in the context window for a thought is answered
directly.
"""

from __future__ import annotations

import math
import uuid
import zlib

from ajev.lm.labels import option_labels
from ajev.lm.predictor import load_tokenizer, prompt_ids, read_lm_config, resolve_adapter, resolve_prompt_version
from ajev.lm.prompt import MAX_OPTIONS
from ajev.lm.reread import DEFAULT_THRESHOLD, combine, needs_reread, parse_reread, reversed_decision
from ajev.lm.thinking import (ThinkingConfig, ThoughtInfo, close_token_id, end_token_ids, is_unsure, parse_mode,
                              readout_ids, thought_room, thought_text)
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
                 temperatures: dict[str, float] | None = None, thinking: ThinkingConfig | None = None,
                 prompt_version: str | None = None, reread: str | None = None,
                 reread_threshold: float | None = None) -> None:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        self.SamplingParams = SamplingParams
        self.tok = load_tokenizer(model_id)
        self.max_model_len = max_model_len
        cfg = read_lm_config(adapter)
        self.prompt_version = resolve_prompt_version(cfg, prompt_version)
        # 重读取平均（ajev/lm/reread.py）：显式参数优先，否则读适配器配置里的 inference.reread，默认 off
        inf = cfg.get("inference") or {}
        self.reread = parse_reread(reread or inf.get("reread"))
        self.reread_threshold = reread_threshold if reread_threshold is not None else \
            float(inf.get("reread_threshold", DEFAULT_THRESHOLD))
        self.temperatures = temperatures if temperatures is not None else cfg.get("temperatures", {})
        self.thinking = thinking or ThinkingConfig.from_lm_config(cfg)
        # 每个适配器路径一个 id：同一个 vLLM 进程里换了适配器（例如不同的 λ），不会用到缓存里的旧权重
        self.lora = LoRARequest(f"ajev-{zlib.crc32(adapter.encode()):08x}", zlib.crc32(adapter.encode()) % (2 ** 31 - 1) + 1,
                                adapter) if adapter else None
        _, ids, valid = option_labels(self.tok)
        self.forms = [[ids[i][j] for j in range(2) if valid[i][j]] for i in range(len(ids))]
        self.end_ids = end_token_ids(self.tok)
        try:
            self.close_id: int | None = close_token_id(self.tok)
        except ValueError:
            self.close_id = None  # this base model has no thinking format; only "off" is served

    def prompt(self, d: Decision) -> list[int]:
        ids = prompt_ids(self.tok, d, NO_TRUNCATION, version=self.prompt_version)
        if len(ids) + 1 > self.max_model_len:
            raise ContextTooLong(f"prompt of {len(ids)} tokens exceeds the maximum context length {self.max_model_len}")
        return ids

    def params(self, d: Decision):
        if len(d.options) > len(self.forms):
            raise ValueError(f"{len(d.options)} options per choice > {len(self.forms)} labels")
        allowed = sorted({t for f in self.forms[: len(d.options)] for t in f})
        return self.SamplingParams(max_tokens=1, temperature=0, logprobs=len(allowed), allowed_token_ids=allowed)

    def check_mode(self, mode) -> str:
        mode = parse_mode(mode)
        if mode != "off" and self.close_id is None:
            raise ValueError("thinking is not supported by this base model")
        return mode

    def think_prompt(self, d: Decision) -> tuple[list[int], int]:
        """(prompt ids that open a thought, thought tokens that fit); 0 tokens means answer directly."""
        ids = prompt_ids(self.tok, d, NO_TRUNCATION, thinking=True, version=self.prompt_version)
        return ids, thought_room(len(ids), self.thinking.budget, self.max_model_len)

    def think_params(self, room: int):
        return self.SamplingParams(max_tokens=room, temperature=0, stop_token_ids=[self.close_id], detokenize=False)

    def finish_thought(self, ids: list[int], output) -> tuple[list[int], ThoughtInfo]:
        """Readout ids (prompt + closed thought) and what the thought did."""
        thought = list(output.outputs[0].token_ids)
        ro, forced = readout_ids(ids, thought, self.close_id, self.end_ids)
        return ro, ThoughtInfo(used=True, tokens=len(thought), forced_close=forced,
                               text=thought_text(self.tok, thought, self.close_id))

    def probs(self, d: Decision, output, thought: bool = False) -> list[float]:
        lp = output.outputs[0].logprobs[0]  # {token_id: Logprob} over the allowed labels
        scores = []
        for f in self.forms[: len(d.options)]:
            vals = [lp[t].logprob for t in f if t in lp] or [-1e4]
            m = max(vals)
            scores.append(m + math.log(sum(math.exp(v - m) for v in vals)))
        t = self.thinking.temperature(d.type, self.temperatures) if thought else self.temperatures.get(d.type, 1.0)
        m = max(scores)
        e = [math.exp((s - m) / t) for s in scores]
        z = sum(e)
        return [x / z for x in e]


class VLLMPredictor:
    """Offline batch inference with the LMPredictor interface.

    ``predict(decisions, thinking="auto")`` also works offline (for example to pick the auto threshold on a
    dev set); what thinking did per decision is left in ``last_thoughts``.
    """

    def __init__(self, model_id: str, adapter: str | None = None, max_model_len: int = 131072,
                 gpu_memory_utilization: float = 0.88, temperatures: dict[str, float] | None = None,
                 thinking: ThinkingConfig | None = None, prompt_version: str | None = None,
                 reread: str | None = None, reread_threshold: float | None = None) -> None:
        from vllm import LLM

        adapter = resolve_adapter(adapter)
        self.reader = LabelReader(model_id, adapter, max_model_len, temperatures, thinking, prompt_version,
                                  reread, reread_threshold)
        self.last_reread: list[bool] = []
        self.tok, self.temperatures = self.reader.tok, self.reader.temperatures
        self.llm = LLM(**engine_kwargs(model_id, adapter, max_model_len, gpu_memory_utilization))
        self.last_thoughts: list[ThoughtInfo] = []

    def _read(self, decisions: list[Decision], ids: list[list[int]], thought: bool) -> list[list[float]]:
        from vllm.inputs import TokensPrompt

        r = self.reader
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids], [r.params(d) for d in decisions],
                                 lora_request=r.lora, use_tqdm=False)
        return [r.probs(d, o, thought) for d, o in zip(decisions, outs)]

    def predict(self, decisions: list[Decision], thinking: str | None = None) -> list[list[float]]:
        from vllm.inputs import TokensPrompt

        r = self.reader
        mode = r.check_mode(thinking)
        probs: list[list[float] | None] = [None] * len(decisions)
        self.last_thoughts = [ThoughtInfo() for _ in decisions]
        self.last_reread = [False] * len(decisions)
        if mode in ("off", "auto"):
            probs = self._read(decisions, [r.prompt(d) for d in decisions], thought=False)
            again = [i for i, d in enumerate(decisions) if needs_reread(r.reread, d, probs[i], r.reread_threshold)]
            if again:  # 倒序再读一次，对数平均（思考之前做，auto 模式按平均后的置信度决定要不要思考）
                rev = [reversed_decision(decisions[i]) for i in again]
                for i, p2 in zip(again, self._read(rev, [r.prompt(d) for d in rev], thought=False)):
                    probs[i], self.last_reread[i] = combine(probs[i], p2), True
        todo = [i for i in range(len(decisions))
                if mode == "on" or is_unsure(probs[i], r.thinking.auto_threshold)]
        plans = [(i, *r.think_prompt(decisions[i])) for i in todo]
        plans = [p for p in plans if p[2] > 0]
        if plans:
            outs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for _, ids, _ in plans],
                                     [r.think_params(room) for _, _, room in plans], lora_request=r.lora,
                                     use_tqdm=False)
            readouts = []
            for (i, ids, _), o in zip(plans, outs):
                ro, self.last_thoughts[i] = r.finish_thought(ids, o)
                readouts.append(ro)
            for (i, _, _), p in zip(plans, self._read([decisions[i] for i, _, _ in plans], readouts, thought=True)):
                probs[i] = p
        missing = [i for i, p in enumerate(probs) if p is None]  # "on" without room for a thought
        if missing:
            for i, p in zip(missing, self._read([decisions[i] for i in missing],
                                                [r.prompt(decisions[i]) for i in missing], thought=False)):
                probs[i] = p
        return probs


class AsyncVLLMPredictor:
    """Async inference for the HTTP server."""

    def __init__(self, model_id: str, adapter: str | None = None, max_model_len: int = 131072,
                 gpu_memory_utilization: float = 0.88, thinking: ThinkingConfig | None = None,
                 prompt_version: str | None = None, reread: str | None = None,
                 reread_threshold: float | None = None) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        adapter = resolve_adapter(adapter)
        self.reader = LabelReader(model_id, adapter, max_model_len, thinking=thinking, prompt_version=prompt_version,
                                  reread=reread, reread_threshold=reread_threshold)
        self.tok, self.temperatures = self.reader.tok, self.reader.temperatures
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs(model_id, adapter, max_model_len,
                                                                                 gpu_memory_utilization)))

    async def _run(self, ids: list[int], params):
        from vllm.inputs import TokensPrompt

        final = None
        async for out in self.engine.generate(TokensPrompt(prompt_token_ids=ids), params,
                                              request_id=uuid.uuid4().hex, lora_request=self.reader.lora):
            final = out
        return final

    async def _one(self, d: Decision, ids: list[int], thought: bool = False) -> list[float]:
        return self.reader.probs(d, await self._run(ids, self.reader.params(d)), thought)

    async def _think(self, d: Decision, direct: list[float] | None) -> tuple[list[float], ThoughtInfo, int]:
        """Think, then read the labels: (probabilities, thought info, prompt tokens sent).
        Without room for a thought the direct answer is kept (computed here if not given)."""
        r = self.reader
        ids, room = r.think_prompt(d)
        if room == 0:
            if direct is not None:
                return direct, ThoughtInfo(), 0
            ids = r.prompt(d)
            return await self._one(d, ids), ThoughtInfo(), len(ids)
        ro, info = r.finish_thought(ids, await self._run(ids, r.think_params(room)))
        return await self._one(d, ro, thought=True), info, len(ids)

    async def predict(self, decisions: list[Decision], thinking: str | None = None) \
            -> tuple[list[list[float]], int, list[ThoughtInfo]]:
        """(probabilities, prompt tokens sent, thought info) per request; questions are submitted together.

        thinking: "off" (default), "on" or "auto" (see ajev/lm/thinking.py).
        """
        import asyncio

        r = self.reader
        mode = r.check_mode(thinking)
        infos = [ThoughtInfo() for _ in decisions]
        if mode == "on":
            res = await asyncio.gather(*(self._think(d, None) for d in decisions))
            return [p for p, _, _ in res], sum(n for _, _, n in res), [i for _, i, _ in res]
        ids = [r.prompt(d) for d in decisions]
        probs = list(await asyncio.gather(*(self._one(d, i) for d, i in zip(decisions, ids))))
        tokens = sum(len(i) for i in ids)
        again = [k for k, d in enumerate(decisions) if needs_reread(r.reread, d, probs[k], r.reread_threshold)]
        if again:  # 倒序再读一次，对数平均
            rev = [reversed_decision(decisions[k]) for k in again]
            rev_ids = [r.prompt(d) for d in rev]
            p2s = await asyncio.gather(*(self._one(d, i) for d, i in zip(rev, rev_ids)))
            for k, p2 in zip(again, p2s):
                probs[k] = combine(probs[k], p2)
            tokens += sum(len(i) for i in rev_ids)
        if mode == "auto":
            unsure = [k for k, p in enumerate(probs) if is_unsure(p, r.thinking.auto_threshold)]
            res = await asyncio.gather(*(self._think(decisions[k], probs[k]) for k in unsure))
            for k, (p, info, n) in zip(unsure, res):
                probs[k], infos[k] = p, info
                tokens += n
        return probs, tokens, infos
