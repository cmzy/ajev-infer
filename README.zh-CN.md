# ajev-infer

[English](README.md) · **简体中文**

**AJev** 的推理代码。AJev 是仿照 Jev 的**带类型决策模型**：输入一段材料（`state`，文本或 JSON）和若干带类型的问题：

- 是 / 否（`noul`）；
- 单选（`choice`，最多 255 个选项）；
- 按顺序排列的等级（`score`）。

模型给每个选项输出**校准过的概率**。每个模型都由一个 Gemma 4 底座和一个 LoRA 适配器组成：

| 适配器 | 底座 | Decision Index 0.2.1 | bf16 显存 |
|---|---|---:|---:|
| [andyzhang232/ajev-gemma4-26b-a4b-lora1](https://huggingface.co/andyzhang232/ajev-gemma4-26b-a4b-lora1)（推荐） | `google/gemma-4-26B-A4B-it` | 57.42 | 约 55 GB |
| [andyzhang232/ajev-gemma4-12b-lora7](https://huggingface.co/andyzhang232/ajev-gemma4-12b-lora7) | `google/gemma-4-12B-it` | 55.06 | 约 24 GB |

成绩和训练细节见各自的模型卡。下面的例子用的是 12B 适配器；用 26B-A4B 时，把底座换成 `google/gemma-4-26B-A4B-it`
（`LMPredictor("google/gemma-4-26B-A4B-it", adapter=...)`，或 `serve_vllm --base-model google/gemma-4-26B-A4B-it`）。

这个仓库只包含推理代码，导入时的包名是 `ajev`。

## 安装

```sh
# transformers 版（CUDA、Apple MPS 或 CPU）
pip install "ajev-infer @ git+https://github.com/cmzy/ajev-infer"
# 加装 vLLM 服务（NVIDIA 显卡）
pip install "ajev-infer[vllm] @ git+https://github.com/cmzy/ajev-infer"
```

**注意：**
- 要用 transformers 5.17 或更新的版本，更早的版本对 Gemma 4 的分词结果和训练时不同。
- 基座模型 `google/gemma-4-12B-it` 在 bf16 下约 24 GB。
- 适配器约 0.5 GB，第一次使用时自动从 HuggingFace 下载。

## Python 调用

```python
from ajev.lm.predictor import LMPredictor
from ajev.schema import decisions_from_jev, jev_answer

p = LMPredictor("google/gemma-4-12B-it", adapter="andyzhang232/ajev-gemma4-12b-lora7")
state = {"工单": "订单 #1182 被扣了两次款，发邮件也没人回。", "会员等级": "金卡"}
questions = {
    "类别": {"type": "choice", "instructions": "这张工单属于哪类问题？",
             "criteria": {"账单": "扣费、发票、退款", "物流": "发货和配送", "账户": "登录和设置"}},
    "转人工": {"type": "noul", "instructions": "需要马上转给人工客服。"},
    "情绪": {"type": "score", "instructions": "客户有多生气？", "criteria": ["平静", "不满", "生气", "非常愤怒"]},
}
ds = decisions_from_jev(state, questions)
for d, probs in zip(ds, p.predict(ds)):
    print(d.meta["question_id"], jev_answer(d, probs))
```

`LMPredictor(..., prefix_cache=True)` 会让同一请求里的多个问题共用材料部分的计算，问题多时更快。适配器里的校准温度会自动读取并使用。

## HTTP 服务（vLLM）

```sh
python -m ajev.serve_vllm --adapter andyzhang232/ajev-gemma4-12b-lora7 --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' \
  -d '{"state": "包裹 10 月 2 日已出库。", "questions": {"已发货": {"type": "noul", "instructions": "包裹已经发出了。"}}}'
```

- `POST /v1/systemone` 的请求和响应都是 Jev 格式，同时到达的请求会由 vLLM 合并成批次一起算。
- 提示词超过 `--max-model-len`（默认 131,072 个 token）时，返回 HTTP 422，不会截断材料。
- 适配器直接作为 LoRA 加载。不要使用“合并后保存”的模型：在 transformers 5.17 下，Gemma 4 合并后保存的模型重新加载后，结果会不一样。

## Decision Index 排行榜

装好[排行榜复现工具包](https://github.com/apolinario/decision-index)并构建好题库后：

```sh
pip install "ajev-infer[leaderboard] @ git+https://github.com/cmzy/ajev-infer"
# 进程内运行（transformers），一次一个请求，排行榜测延迟用的就是这种方式
python -m decision_index run --engine ajev.jdi_engine:AJevEngine \
    --option adapter=andyzhang232/ajev-gemma4-12b-lora7 --out runs/ajev
# 或者连上面的 vLLM 服务
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=ajev --out runs/ajev
python -m decision_index score --results runs/ajev/results.jsonl
```

两种引擎都会把每个请求完整发送，不截断、不删选项，提示词也和平时使用时完全相同。

## 打分原理

1. 每个问题写成一段对话提示，依次是：说明、材料、问题，以及标为 `A`、`B`… 的选项。超过 `Z` 之后用单 token 的两字母编码，最多 255 个选项。
2. 模型读一遍提示词，取下一个 token 位置上各个选项标签的 logits。
3. 按题型除以在留出数据上拟合的温度，再做 softmax 得到概率。

整个过程不生成任何文字。

## 许可证

Apache-2.0。适配器的训练数据来自多个公开数据集，它们各有自己的许可条款，其中有些是非商业许可，详见模型卡。
