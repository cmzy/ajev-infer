"""Thinking modes: answer directly (off, the default), think first (on), or think only when unsure (auto).

    off   one forward pass reads the option-label logits (the original behaviour).
    on    the model first writes a thought, greedily, up to ``budget`` tokens; then the label logits are read
          at the answer position right after the closed thought.
    auto  answer directly first; questions whose top option probability is below ``auto_threshold`` are
          answered again with thinking.

给初学者的说明：
- **思考在提示词里是什么样子**：聊天模板有一个 ``enable_thinking`` 开关。关闭时（我们训练和平时推理都用这种），
  Gemma 4 的提示词结尾是 ``<|turn>model\\n<|channel>thought\\n<channel|>``——一段已经“关上”的空思考，
  下一个 token 就是答案字母。打开时，模板在开头加一个 ``<|think|>`` 标记，模型会自己写出
  ``<|channel>thought\\n……<channel|>``。Qwen 系列对应的是 ``<think>……</think>``。
- **怎么从思考得到概率**：模型写完思考（遇到“关闭标记”就停下）后，我们把“提示词 + 思考”拼起来再算一次，
  读答案位置上各个选项字母的打分，和不思考时完全同一种读法，所以返回的仍然是每个选项的概率。
- **预算**：思考最多写 ``budget`` 个 token。写满了还没关闭，就替模型补上关闭标记，强制它给答案——
  这样延迟有上限，也不会出现“永远想不完”的题。
- **贪心解码（temperature=0）**：同样的请求每次想出同样的内容，结果可以复现。
- **auto 的门槛**：最高概率低于门槛才思考。门槛只能在自己的开发集上定，不能拿排行榜题目来调。

本模块不依赖 torch，vLLM 和 transformers 两种推理后端共用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

MODES = ("off", "on", "auto")
DEFAULT_MODE = "off"
DEFAULT_BUDGET = 1024
DEFAULT_AUTO_THRESHOLD = 0.6
# With less room than this left in the context window, the question is answered without thinking.
MIN_BUDGET = 16
# Tokens that close a thought, by chat-template family (Gemma 4, Qwen 3.x).
CLOSE_MARKERS = ("<channel|>", "</think>")
# End-of-turn markers: generation can stop on one of these before the thought is closed.
END_MARKERS = ("<turn|>", "<end_of_turn>", "<|im_end|>", "<|endoftext|>", "<eos>")


def parse_mode(value, default: str = DEFAULT_MODE) -> str:
    """"off" / "on" / "auto" (any case); None gives the default; True / False mean on / off."""
    if value is None:
        return default
    if value is True or value is False:
        return "on" if value else "off"
    if isinstance(value, str) and value.strip().lower() in MODES:
        return value.strip().lower()
    raise ValueError(f"thinking must be one of {', '.join(MODES)}; got {value!r}")


@dataclass
class ThinkingConfig:
    """Thinking settings. The adapter's config file may carry them under "thinking"; the server flags override.

    temperatures: per-type temperatures for answers after thinking; empty means the no-thinking ones are used.
    """

    budget: int = DEFAULT_BUDGET
    auto_threshold: float = DEFAULT_AUTO_THRESHOLD
    temperatures: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_lm_config(cls, cfg: dict, budget: int | None = None, auto_threshold: float | None = None) -> "ThinkingConfig":
        sec = cfg.get("thinking") or {}
        c = cls(budget=int(sec.get("budget", DEFAULT_BUDGET)),
                auto_threshold=float(sec.get("auto_threshold", DEFAULT_AUTO_THRESHOLD)),
                temperatures=dict(sec.get("temperatures", {})))
        if budget is not None:
            c.budget = budget
        if auto_threshold is not None:
            c.auto_threshold = auto_threshold
        if c.budget < 1:
            raise ValueError(f"thinking budget must be at least 1 token; got {c.budget}")
        if not 0.0 <= c.auto_threshold <= 1.0:
            raise ValueError(f"auto threshold must be in [0, 1]; got {c.auto_threshold}")
        return c

    def temperature(self, qtype: str, fallback: dict[str, float]) -> float:
        return self.temperatures.get(qtype, fallback.get(qtype, 1.0))


def close_token_id(tok) -> int:
    """Id of the single token that closes a thought; ValueError if the tokenizer has none."""
    vocab = tok.get_vocab()
    for marker in CLOSE_MARKERS:
        if marker in vocab:
            return vocab[marker]
    raise ValueError(f"thinking is not supported for this tokenizer (no {' / '.join(CLOSE_MARKERS)} token)")


def end_token_ids(tok) -> set[int]:
    """Ids of the end-of-sequence / end-of-turn tokens of this tokenizer."""
    vocab = tok.get_vocab()
    ids = {vocab[m] for m in END_MARKERS if m in vocab}
    for t in (tok.eos_token_id, tok.pad_token_id):
        if t is not None:
            ids.add(t)
    return ids


def chat_ids(tok, user_message: str | list[dict], thinking: bool) -> list[int]:
    """Chat-template token ids ending at the generation prompt, with thinking switched on or off.

    ``user_message`` is one user message, or a full message list (e.g. with a system message, see ajev/lm/prompt.py).
    Gemma 4's template defaults to off and Qwen 3.x's to on, so the switch is always passed explicitly.
    """
    messages = [{"role": "user", "content": user_message}] if isinstance(user_message, str) else user_message
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=thinking)
    if not isinstance(ids, list):  # some transformers versions return a dict
        ids = ids["input_ids"]
    return list(ids)


def thought_room(prompt_len: int, budget: int, max_len: int) -> int:
    """Thought tokens that fit: the budget, minus what the context window cannot hold
    (one place each for a forced close token and the answer label). 0 means answer without thinking."""
    room = min(budget, max_len - prompt_len - 2)
    return room if room >= MIN_BUDGET else 0


def readout_ids(prompt: list[int], thought: list[int], close_id: int,
                end_ids: set[int] = frozenset()) -> tuple[list[int], bool]:
    """Prompt + thought, ending at the answer position; (ids, whether the close token had to be added).

    The generated thought may or may not include the close token, depending on the backend. Anything after
    the first close token is dropped: the answer is read right after the close. A thought that ended the
    turn (or hit the budget) without closing has its trailing end tokens removed and the close token added.
    """
    if close_id in thought:
        return prompt + thought[: thought.index(close_id) + 1], False
    thought = list(thought)
    while thought and thought[-1] in end_ids:
        thought.pop()
    return prompt + thought + [close_id], True


def is_unsure(probs: list[float], threshold: float) -> bool:
    """True when the top option probability is below the threshold (auto mode thinks on these)."""
    return max(probs) < threshold


@dataclass
class ThoughtInfo:
    """What thinking did for one question; returned to the caller alongside the probabilities."""

    used: bool = False
    tokens: int = 0
    forced_close: bool = False
    text: str | None = None

    def to_json(self, include_text: bool) -> dict:
        out = {"used": self.used}
        if self.used:
            out.update(tokens=self.tokens, forced_close=self.forced_close)
            if include_text and self.text is not None:
                out["text"] = self.text
        return out


def thought_text(tok, thought: list[int], close_id: int) -> str:
    """Readable thought: the generated tokens up to the close token, special tokens removed."""
    if close_id in thought:
        thought = thought[: thought.index(close_id)]
    text = tok.decode(thought, skip_special_tokens=True).strip()
    # Gemma 4 opens the thought with "<|channel>thought\n"; without special tokens the word "thought" remains.
    return text[len("thought"):].lstrip() if text.startswith("thought\n") else text
