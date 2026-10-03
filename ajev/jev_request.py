"""Normalise Jev / Decision Index requests (shared by the HTTP server and the leaderboard engine)."""

from __future__ import annotations

import json
import re

CJK = re.compile(r"[一-鿿]")


def text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def detect_lang(state, questions: dict) -> str:
    """"zh" if more than 10% of the state and instructions are CJK characters, else "en"."""
    s = text(state) + "".join(text(q.get("instructions", "")) for q in questions.values())
    return "zh" if s and len(CJK.findall(s)) / len(s) > 0.1 else "en"


def plain_questions(questions: dict) -> dict:
    """Turn JSON instructions and option descriptions into strings."""
    out = {}
    for k, q in questions.items():
        q = dict(q)
        q["instructions"] = text(q.get("instructions", ""))
        if q["type"] == "choice":
            q["criteria"] = {name: "" if desc is None else text(desc) for name, desc in q["criteria"].items()}
        elif q.get("criteria") and isinstance(q["criteria"], dict):
            q["criteria"] = {name: text(desc) for name, desc in q["criteria"].items()}
        out[k] = q
    return out
