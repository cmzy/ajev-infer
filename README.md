# ajev-infer

**English** · [简体中文](README.zh-CN.md)

Inference code for **AJev**, a Jev-style *typed decision model*: give it a `state` (any text or JSON) and typed
questions — yes/no (`noul`), single choice (`choice`, up to 255 options) or ordered levels (`score`) — and it returns a
calibrated probability for every option. Each model is a Gemma 4 base plus a LoRA adapter:

| Adapter | Base | Decision Index 0.2.1 | bf16 memory |
|---|---|---:|---:|
| [andyzhang232/ajev-gemma4-26b-a4b-lora1](https://huggingface.co/andyzhang232/ajev-gemma4-26b-a4b-lora1) (recommended) | `google/gemma-4-26B-A4B-it` | 57.42 | ~55 GB |
| [andyzhang232/ajev-gemma4-12b-lora7](https://huggingface.co/andyzhang232/ajev-gemma4-12b-lora7) | `google/gemma-4-12B-it` | 55.06 | ~24 GB |

This is the inference code only; the import name is `ajev`. Results and training details are in the model cards.
The examples below use the 12B adapter; for the 26B-A4B one, pass `google/gemma-4-26B-A4B-it` as the base model
(`LMPredictor("google/gemma-4-26B-A4B-it", adapter=...)`, or `serve_vllm --base-model google/gemma-4-26B-A4B-it`).

## Install

```sh
# transformers backend (CUDA, Apple MPS or CPU)
pip install "ajev-infer @ git+https://github.com/cmzy/ajev-infer"
# + vLLM server (NVIDIA GPUs)
pip install "ajev-infer[vllm] @ git+https://github.com/cmzy/ajev-infer"
```

Use transformers 5.17 or newer: older versions tokenize Gemma 4 differently from training. The base model is
`google/gemma-4-12B-it` (about 24 GB in bf16); the adapter is 0.5 GB and is downloaded from the Hub on first use.

## Python

```python
from ajev.lm.predictor import LMPredictor
from ajev.schema import decisions_from_jev, jev_answer

p = LMPredictor("google/gemma-4-12B-it", adapter="andyzhang232/ajev-gemma4-12b-lora7")
state = {"ticket": "I was charged twice for order #1182 and nobody answers my emails.", "tier": "gold"}
questions = {
    "topic":  {"type": "choice", "instructions": "What is the ticket about?",
               "criteria": {"billing": "charges, invoices, refunds", "delivery": "shipping and tracking",
                            "account": "login and settings"}},
    "escalate": {"type": "noul", "instructions": "Should this go to a human agent right away?"},
    "anger":  {"type": "score", "instructions": "How upset is the customer?",
               "criteria": ["calm", "annoyed", "angry", "furious"]},
}
ds = decisions_from_jev(state, questions)
for d, probs in zip(ds, p.predict(ds)):
    print(d.meta["question_id"], jev_answer(d, probs))
```

`LMPredictor(..., prefix_cache=True)` computes the shared state once for all questions of a request (faster when a
request has several questions). Calibration temperatures are read from the adapter and applied automatically.

## HTTP server (vLLM)

```sh
python -m ajev.serve_vllm --adapter andyzhang232/ajev-gemma4-12b-lora7 --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' \
  -d '{"state": "The parcel left the warehouse on 2 Oct.", "questions": {"shipped": {"type": "noul", "instructions": "Has the parcel shipped?"}}}'
```

`POST /v1/systemone` takes and returns the Jev wire format; concurrent requests are batched by vLLM. Prompts longer
than `--max-model-len` (default 131,072 tokens) are refused with HTTP 422 instead of being truncated. The adapter is
loaded directly as a LoRA; do not serve a saved merged checkpoint (with transformers 5.17 a merged Gemma 4 checkpoint
reloads with different outputs).

## Decision Index (leaderboard)

With the [reproduction kit](https://github.com/apolinario/decision-index) installed and its suite built:

```sh
pip install "ajev-infer[leaderboard] @ git+https://github.com/cmzy/ajev-infer"
# in-process (transformers), one request at a time — the setting the board uses for latency
python -m decision_index run --engine ajev.jdi_engine:AJevEngine \
    --option adapter=andyzhang232/ajev-gemma4-12b-lora7 --out runs/ajev
# or against the vLLM server above
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=ajev --out runs/ajev
python -m decision_index score --results runs/ajev/results.jsonl
```

Both engines send every request whole (no truncation, no option pruning) with the same prompt as everyday use.

## How it scores

Each question becomes one chat prompt: instructions, the state, the question and the options labelled `A`, `B`, …
(after `Z`: single-token two-letter codes, up to 255 options). The model reads the prompt once and the next-token
logits of the option labels are softmaxed, divided by a per-question-type temperature fitted on held-out data. No
text is generated.

## License

Apache-2.0. The adapter is trained on a mixture of public datasets with their own terms (some are non-commercial);
see the model card.
