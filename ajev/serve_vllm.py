"""Jev-compatible HTTP server on vLLM: ``POST /v1/systemone``.

    python -m ajev.serve_vllm --adapter andyzhang232/ajev-gemma4-12b-lora5 --port 8000
    python -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=ajev

Concurrent requests are batched by vLLM. Prompts longer than --max-model-len get HTTP 422
("maximum context length"); nothing is truncated. Every answer carries its "type".
"""

from __future__ import annotations

import argparse
import time

from ajev.jev_request import detect_lang, plain_questions
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


def create_app(predictor: AsyncVLLMPredictor, model_name: str):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="AJev (vLLM)")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": model_name, "temperatures": predictor.temperatures,
                "max_model_len": predictor.reader.max_model_len}

    @app.post("/v1/systemone")
    async def systemone(body: dict) -> dict:
        questions = body.get("questions")
        if "state" not in body or not isinstance(questions, dict) or not questions:
            raise HTTPException(400, "request needs 'state' and a non-empty 'questions' object")
        if len(questions) > MAX_QUESTIONS:
            raise HTTPException(422, f"too many questions: {len(questions)} > {MAX_QUESTIONS}")
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
            probs, input_tokens = await predictor.predict(decisions)
        except ContextTooLong as e:
            raise HTTPException(422, str(e)) from e
        except ValueError as e:  # more options than labels
            raise HTTPException(422, str(e)) from e
        return {"model": model_name, "answers": {d.meta["question_id"]: answer(d, p) for d, p in zip(decisions, probs)},
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
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
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    predictor = AsyncVLLMPredictor(a.base_model, a.adapter, a.max_model_len, a.gpu_memory_utilization)
    uvicorn.run(create_app(predictor, a.name), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
