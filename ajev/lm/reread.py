"""Reread: ask a question a second time with the options in reverse order and average the two readings in log space.

- **Why**: models have position preferences (e.g. favouring the first or the last option). The reversed reading has
  the opposite position bias, so averaging cancels part of it; it matters most on uncertain questions.
- **Log-space average**: log p1 and log p2 are averaged and renormalised (a geometric mean), which is less sensitive
  to one extreme reading than averaging the probabilities.
- **Score questions are never reversed**: their levels are ordered (low to high), and reversing would change the
  question.

Modes: ``off`` (read once, the default), ``unsure`` (read again when the top probability is below the threshold),
``all`` (read every question twice).
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
    """A copy with options and target reversed (the id gets a suffix so it is never mixed up with the original)."""
    return Decision(id=d.id + "#rev", source=d.source, type=d.type, state=d.state, instructions=d.instructions,
                    options=list(reversed(d.options)), target=list(reversed(d.target)), lang=d.lang, group=d.group,
                    meta=d.meta)


def combine(p1: list[float], p2_reversed: list[float]) -> list[float]:
    """Combine the first reading with the reversed one (given in reversed option order): log-space mean, renormalised."""
    p2 = list(reversed(p2_reversed))
    logs = [(math.log(max(a, 1e-12)) + math.log(max(b, 1e-12))) / 2 for a, b in zip(p1, p2)]
    m = max(logs)
    e = [math.exp(x - m) for x in logs]
    z = sum(e)
    return [x / z for x in e]
