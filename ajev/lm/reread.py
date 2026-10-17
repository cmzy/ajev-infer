"""换一种选项顺序再读一次，两次结果在对数空间取平均（docs/plan_lm6.md 第 10.3 节，第 0⁗ 步）。

给初学者的说明：
- **为什么有用**：模型对选项的位置有偏好（比如偏爱第一个或最后一个）。把选项倒过来再问一次，两次的位置偏好
  方向相反，取平均后能抵消一部分；对拿不准的题效果最明显。decider-31b 对最高概率低于 0.7 的题（约 28%）这样做，
  anyjev-l0 对所有题用两种轮换顺序。
- **对数空间取平均**：两次概率 p1、p2 先取对数、相加除以 2，再归一化（几何平均），比直接平均概率更不容易被
  某一次的极端值带偏。
- **打分题（score）不倒序**：等级是有顺序的（低 → 高），倒过来会改变题意。

模式：``off``（只读一次，默认）、``unsure``（最高概率低于门槛的题再读一次）、``all``（每道题都读两次）。
"""

from __future__ import annotations

import math

from ajev.schema import Decision

REREAD_MODES = ("off", "unsure", "all")
DEFAULT_THRESHOLD = 0.7


def parse_reread(mode: str | None) -> str:
    mode = (mode or "off").strip().lower()
    if mode not in REREAD_MODES:
        raise ValueError(f"reread must be one of {', '.join(REREAD_MODES)}; got {mode!r}")
    return mode


def needs_reread(mode: str, d: Decision, probs: list[float], threshold: float = DEFAULT_THRESHOLD) -> bool:
    if mode == "off" or d.type == "score":
        return False
    return mode == "all" or max(probs) < threshold


def reversed_decision(d: Decision) -> Decision:
    """选项和 target 都倒序的副本（id 加后缀，避免和原题混在一起）。"""
    return Decision(id=d.id + "#rev", source=d.source, type=d.type, state=d.state, instructions=d.instructions,
                    options=list(reversed(d.options)), target=list(reversed(d.target)), lang=d.lang, group=d.group,
                    meta=d.meta)


def combine(p1: list[float], p2_reversed: list[float]) -> list[float]:
    """第一次的概率和倒序那次的概率（按倒序后的选项顺序）合并：对数平均，再归一化。"""
    p2 = list(reversed(p2_reversed))
    logs = [(math.log(max(a, 1e-12)) + math.log(max(b, 1e-12))) / 2 for a, b in zip(p1, p2)]
    m = max(logs)
    e = [math.exp(x - m) for x in logs]
    z = sum(e)
    return [x / z for x in e]
