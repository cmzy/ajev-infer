"""Prompt format: one chat message per decision, options labelled A, B, C, ...

The model is never asked to generate text; the next-token logits of the labels are read instead.
The state comes first so that questions on the same state share a prompt prefix (and its KV cache).

    You are a decision model. Read the state and answer the question by picking exactly one option.

    ### State
    {state}

    ### Question
    {instructions}
    {type hint}

    ### Options
    A. delivery: shipping
    B. refund: money back

    Reply with the letter of the best option only.
"""

from __future__ import annotations

from ajev.schema import Decision

LETTERS = [chr(ord("A") + i) for i in range(26)]
# Beyond 26 options, labels continue with single-token two-letter codes (see ajev/lm/labels.py).
MAX_OPTIONS = 255
MAX_LETTER_OPTIONS = len(LETTERS)

TYPE_HINT = {
    "noul": "Decide whether the statement holds.",
    "choice": "Pick the single best option.",
    "score": "Pick the level that best fits; levels are ordered from lowest to highest.",
}


def option_lines(d: Decision, labels: list[str] | None = None) -> list[str]:
    """``A. name: description`` per option; noul options are shown as Yes / No."""
    lines = []
    for letter, o in zip(labels or LETTERS, d.options):
        name = {"true": "Yes", "false": "No"}.get(o.name, o.name) if d.type == "noul" else o.name
        lines.append(f"{letter}. {name}: {o.desc}" if o.desc else f"{letter}. {name}")
    return lines


def build_user_message(d: Decision, state: str | None = None, labels: list[str] | None = None) -> str:
    """The user message for one decision. Up to 26 options the text says "letter", beyond that "code"."""
    labels = labels or LETTERS
    if len(d.options) > len(labels):
        raise ValueError(f"{d.id}: {len(d.options)} options > {len(labels)} labels")
    word = "letter" if len(d.options) <= MAX_LETTER_OPTIONS else "code"
    return (
        "You are a decision model. Read the state and answer the question by picking exactly one option.\n\n"
        f"### State\n{d.state if state is None else state}\n\n"
        f"### Question\n{d.instructions}\n{TYPE_HINT[d.type]}\n\n"
        "### Options\n" + "\n".join(option_lines(d, labels)) + "\n\n"
        f"Reply with the {word} of the best option only."
    )
