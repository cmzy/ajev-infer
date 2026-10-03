"""Core data structures shared by data building, training, evaluation and serving.

A Jev request is one ``state`` plus several typed questions:

    noul    yes / no; the answer is P(true)
    choice  one of 2..255 named options
    score   one of 2..10 ordered levels

Internally each question becomes one ``Decision`` (state + question + options + target distribution).
``decisions_from_jev`` splits a request into decisions and ``jev_answer`` turns predicted probabilities
back into the Jev answer format.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

DecisionType = Literal["noul", "choice", "score"]
TYPES: tuple[DecisionType, ...] = ("noul", "choice", "score")

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10

NOUL_TRUE, NOUL_FALSE = "true", "false"
# Option descriptions used when a noul question has no criteria.
DEFAULT_NOUL_DESC = {
    "en": {NOUL_TRUE: "Yes, the statement holds.", NOUL_FALSE: "No, the statement does not hold."},
    "zh": {NOUL_TRUE: "是，该陈述成立。", NOUL_FALSE: "否，该陈述不成立。"},
}


@dataclass
class Option:
    """A candidate answer: a unique ``name`` and an optional description."""

    name: str
    desc: str = ""


@dataclass
class Decision:
    """One question over one state; also one line of a JSONL data file.

    ``target`` is a distribution over ``options`` (one-hot for hard labels, soft for multi-rater labels).
    Decisions with the same ``group`` share a state.
    """

    id: str
    source: str
    type: DecisionType
    state: str
    instructions: str
    options: list[Option]
    target: list[float]
    lang: str = "en"
    group: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.options = [o if isinstance(o, Option) else Option(**o) for o in self.options]

    def validate(self) -> None:
        """Raise ValueError if the type, options or target are inconsistent."""
        k = len(self.options)
        if self.type not in TYPES:
            raise ValueError(f"{self.id}: bad type {self.type!r}")
        if self.type == "noul" and [o.name for o in self.options] not in (
            [NOUL_TRUE, NOUL_FALSE],
            [NOUL_FALSE, NOUL_TRUE],
        ):
            raise ValueError(f"{self.id}: noul options must be true/false")
        if self.type == "choice" and not 2 <= k <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"{self.id}: choice needs 2..{MAX_CHOICE_OPTIONS} options, got {k}")
        if self.type == "score" and not MIN_SCORE_LEVELS <= k <= MAX_SCORE_LEVELS:
            raise ValueError(f"{self.id}: score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels, got {k}")
        if len({o.name for o in self.options}) != k:
            raise ValueError(f"{self.id}: duplicate option names")
        if len(self.target) != k:
            raise ValueError(f"{self.id}: target has {len(self.target)} entries for {k} options")
        # Soft labels may not sum to exactly 1 after rounding.
        if any(p < 0 for p in self.target) or abs(sum(self.target) - 1) > 1e-3:
            raise ValueError(f"{self.id}: target is not a distribution: {self.target}")

    @property
    def option_names(self) -> list[str]:
        return [o.name for o in self.options]

    @property
    def gold_index(self) -> int:
        """Index of the most probable option in ``target`` (first one on ties)."""
        return max(range(len(self.target)), key=self.target.__getitem__)

    def shuffled(self, rng: random.Random) -> "Decision":
        """Copy with options (and target) in random order; score levels are never shuffled."""
        if self.type == "score":
            return self
        perm = list(range(len(self.options)))
        rng.shuffle(perm)
        return Decision(
            **{**asdict(self), "options": [self.options[i] for i in perm], "target": [self.target[i] for i in perm]}
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["target"] = [round(p, 6) for p in self.target]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Decision":
        return cls(**d)


def one_hot(k: int, i: int) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(k)]


def normalize(ps: Iterable[float]) -> list[float]:
    """Clip negatives to 0 and rescale to sum to 1 (uniform if everything is 0)."""
    ps = [max(0.0, float(p)) for p in ps]
    s = sum(ps)
    return [p / s for p in ps] if s > 0 else [1.0 / len(ps)] * len(ps)


def jev_confidence(probs: list[float]) -> float:
    """Jev's confidence: 1 - H(p) / ln K (1 = certain, 0 = uniform)."""
    k = len(probs)
    if k < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, 1.0 - h / math.log(k))


def read_jsonl(path: str) -> list[Decision]:
    with open(path, encoding="utf-8") as f:
        return [Decision.from_dict(json.loads(line)) for line in f if line.strip()]


def write_jsonl(path: str, decisions: Iterable[Decision]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


# ---- Jev wire format ---------------------------------------------------------


def state_to_text(state: Any) -> str:
    """Strings pass through; JSON values become compact JSON text."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def _options_from_jev_question(q: dict[str, Any], lang: str) -> list[Option]:
    """noul: [true, false] with optional descriptions; choice: {name: desc}; score: level i is named "i"."""
    qtype, criteria = q["type"], q.get("criteria")
    if qtype == "noul":
        criteria = criteria or {}
        return [
            Option(NOUL_TRUE, criteria.get(NOUL_TRUE) or DEFAULT_NOUL_DESC[lang][NOUL_TRUE]),
            Option(NOUL_FALSE, criteria.get(NOUL_FALSE) or DEFAULT_NOUL_DESC[lang][NOUL_FALSE]),
        ]
    if qtype == "choice":
        return [Option(name, desc or "") for name, desc in criteria.items()]
    if qtype == "score":
        return [Option(str(i), level) for i, level in enumerate(criteria)]
    raise ValueError(f"unknown question type {qtype!r}")


def decisions_from_jev(
    state: Any,
    questions: dict[str, dict[str, Any]],
    *,
    group: str = "req",
    source: str = "jev",
    lang: str = "en",
    gold: dict[str, dict[str, Any]] | None = None,
) -> list[Decision]:
    """Split one Jev request into decisions, in question order.

    ``meta["question_id"]`` keeps the original question key. ``gold`` (``{qid: {"probabilities": {...}}}``)
    sets the target; without it the target is uniform.
    """
    text = state_to_text(state)
    out = []
    for qid, q in questions.items():
        options = _options_from_jev_question(q, lang)
        if gold and qid in gold:
            probs = gold[qid]["probabilities"]
            target = normalize(probs.get(o.name, 0.0) for o in options)
        else:
            target = [1.0 / len(options)] * len(options)
        out.append(
            Decision(
                id=f"{group}/{qid}",
                source=source,
                type=q["type"],
                state=text,
                instructions=q.get("instructions", ""),
                options=options,
                target=target,
                lang=lang,
                group=group,
                meta={"question_id": qid},
            )
        )
    return out


def jev_answer(decision: Decision, probs: list[float]) -> dict[str, Any]:
    """Probabilities -> Jev answer: {"noul": p}, {"choice", "probabilities", "confidence"},
    or {"score": expected level, "legend", "probabilities", "confidence"}."""
    names = decision.option_names
    pmap = {n: p for n, p in zip(names, probs)}
    conf = round(jev_confidence(probs), 4)
    if decision.type == "noul":
        return {"noul": round(pmap[NOUL_TRUE], 4)}
    if decision.type == "choice":
        best = names[max(range(len(probs)), key=probs.__getitem__)]
        return {"choice": best, "probabilities": {n: round(p, 4) for n, p in pmap.items()}, "confidence": conf}
    score = sum(i * p for i, p in enumerate(probs))
    return {
        "score": round(score, 4),
        "legend": [o.desc for o in decision.options],
        "probabilities": [round(p, 4) for p in probs],
        "confidence": conf,
    }
