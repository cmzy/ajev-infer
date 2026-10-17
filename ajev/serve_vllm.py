"""Jev-compatible HTTP server on vLLM: ``POST /v1/systemone``.

    python -m ajev.serve_vllm --adapter andyzhang232/ajev-gemma4-12b-lora5 --port 8000
    python -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=ajev

Concurrent requests are batched by vLLM. Prompts longer than --max-model-len get HTTP 422
("maximum context length"); nothing is truncated. Every answer carries its "type".

Thinking (optional request fields; see ajev/lm/thinking.py):
    "thinking": "off" | "on" | "auto"   default "off" (or the server's --thinking)
    "include_thoughts": true            also return the thought text
With thinking "on" or "auto", every answer carries "thinking": {"used", "tokens", "forced_close"[, "text"]}
and usage.output_tokens counts the thought tokens.
"""

from __future__ import annotations

import argparse
import time

from ajev.jev_request import detect_lang, plain_questions
from ajev.lm.predictor import read_lm_config, resolve_adapter
from ajev.lm.prompt import PROMPT_VERSIONS
from ajev.lm.thinking import MODES, ThinkingConfig, parse_mode
from ajev.lm.vllm_predictor import AsyncVLLMPredictor, ContextTooLong
from ajev.schema import NOUL_TRUE, decisions_from_jev, jev_answer

MAX_QUESTIONS = 512


def answer(d, probs: list[float]) -> dict:
    if d.type == "noul":
        return {"type": "noul", "noul": dict(zip(d.option_names, probs))[NOUL_TRUE]}
    if d.type == "choice":
        pmap = dict(zip(d.option_names, probs))
        return {"type": "choice", "choice": max(pmap, key=pmap.get), "probabilities": pmap}
    return {"type": "score", **jev_answer(d, probs)}


def create_app(predictor: AsyncVLLMPredictor, model_name: str, default_thinking: str = "off"):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="AJev (vLLM)")

    @app.get("/health")
    async def health() -> dict:
        r = predictor.reader
        thinking = {"default": default_thinking, "modes": list(MODES) if r.close_id is not None else ["off"],
                    "budget": r.thinking.budget, "auto_threshold": r.thinking.auto_threshold}
        return {"status": "ok", "model": model_name, "temperatures": predictor.temperatures,
                "max_model_len": r.max_model_len, "thinking": thinking}

    @app.post("/v1/systemone")
    async def systemone(body: dict) -> dict:
        questions = body.get("questions")
        if "state" not in body or not isinstance(questions, dict) or not questions:
            raise HTTPException(400, "request needs 'state' and a non-empty 'questions' object")
        if len(questions) > MAX_QUESTIONS:
            raise HTTPException(422, f"too many questions: {len(questions)} > {MAX_QUESTIONS}")
        try:
            mode = predictor.reader.check_mode(parse_mode(body.get("thinking"), default_thinking))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        include = body.get("include_thoughts") is True
        try:
            qs = plain_questions(questions)
            decisions = decisions_from_jev(body["state"], qs, group="request", lang=detect_lang(body["state"], qs))
            for d in decisions:
                d.validate()
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            msg = str(e)
            raise HTTPException(422 if "options" in msg else 400, f"invalid question: {msg}") from e
        t = time.perf_counter()
        try:
            probs, input_tokens, thoughts = await predictor.predict(decisions, thinking=mode)
        except ContextTooLong as e:
            raise HTTPException(422, str(e)) from e
        except ValueError as e:  # more options than labels
            raise HTTPException(422, str(e)) from e
        answers = {}
        for d, p, th in zip(decisions, probs, thoughts):
            answers[d.meta["question_id"]] = a = answer(d, p)
            if mode != "off":
                a["thinking"] = th.to_json(include)
        return {"model": model_name, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": sum(th.tokens for th in thoughts)},
                "latency_ms": round((time.perf_counter() - t) * 1000, 1)}

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="google/gemma-4-12B-it")
    ap.add_argument("--adapter", help="LoRA adapter directory (loaded directly, never merged)")
    ap.add_argument("--name", default="ajev", help="model name reported in responses")
    ap.add_argument("--max-model-len", type=int, default=131072, help="longer prompts are refused, not truncated")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    ap.add_argument("--thinking", choices=MODES, default="off", help="mode for requests that do not set \"thinking\"")
    ap.add_argument("--thinking-budget", type=int, help="max thought tokens (default: adapter config, else 1024)")
    ap.add_argument("--auto-threshold", type=float,
                    help="auto mode thinks when the top probability is below this (default: adapter config, else 0.6)")
    ap.add_argument("--prompt-version", choices=PROMPT_VERSIONS,
                    help="prompt format (default: the adapter's, else v1; see ajev/lm/prompt.py)")
    ap.add_argument("--reread", choices=("off", "unsure", "all"),
                    help="read again with the options reversed and average (default: the adapter's, else off; "
                         "see ajev/lm/reread.py)")
    ap.add_argument("--reread-threshold", type=float, help="unsure = top probability below this (default 0.7)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    adapter = resolve_adapter(a.adapter)
    thinking = ThinkingConfig.from_lm_config(read_lm_config(adapter), a.thinking_budget, a.auto_threshold)
    predictor = AsyncVLLMPredictor(a.base_model, adapter, a.max_model_len, a.gpu_memory_utilization, thinking,
                                   a.prompt_version, a.reread, a.reread_threshold)
    if a.thinking != "off":
        predictor.reader.check_mode(a.thinking)
    uvicorn.run(create_app(predictor, a.name, a.thinking), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
