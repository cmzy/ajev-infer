"""Prompt format: one chat per decision, options labelled A, B, C, ...

Two versions. An adapter records the version it was trained with (``prompt_version`` in its ajev_lm_config.json);
adapters without that field (lora5, lora7, gemma26_lora1) use v1.

v1 (one user message):

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

v2 (2026-10-07, docs/plan_lm6.md section 10.2): **choice questions with at most 26 options** get a short system message (from the public
deck31b / Cygnet recipe) and a plain layout; noul and score questions keep the v1 message. On stock Gemma-4-31B the
new layout reads knowledge questions much better than v1 (MMLU-Pro 53.8 -> 68.3%, AQuA-RAT 55.9 -> 67.7%) and is level
on NLI, intent and stance, but it was worse on score (SST-5 -3.4, also with level numbers kept) and slightly worse on
yes/no (BoolQ -1.0), hence v1 for those two types. Over 26 options (two-letter codes) also stays v1: the system
message asks for a single letter, and only one 60-option set was tested (level), while the suite's widest sets
(CLINC150, BANKING77, API-Bank, ChessBench, POP909) carry about 15% of the public score. Unlike deck31b it keeps option names when they carry meaning
(deck31b shows only descriptions, which breaks label-only options such as NLI or intent names):

    [system] You are a calibration engine. ...
    [user]
    {state}

    {instructions}

    Options:
    A. {description}            name is a bare key such as "A" or "3"
    B. {name}                   no description
    C. {name}: {description}    otherwise

    Answer with the letter of exactly one option, and nothing else:
"""

from __future__ import annotations

import random
import re

from ajev.schema import Decision

LETTERS = [chr(ord("A") + i) for i in range(26)]
# Beyond 26 options, labels continue with single-token two-letter codes (see ajev/lm/labels.py).
MAX_OPTIONS = 255
MAX_LETTER_OPTIONS = len(LETTERS)

PROMPT_VERSIONS = ("v1", "v2", "v3")
SYSTEM_V2 = ("You are a calibration engine. You never answer in prose. You are given a state, a question and "
             "a numbered set of options, and you choose exactly one option. You reply with that option's "
             "LETTER and nothing else — a single character, no words, no punctuation, no explanation.")
# System message used only when thinking: the v2/v3 system message says "never answer in prose", which contradicts
# "reason first, then answer". With thinking this one is used instead, and the user message is not repeated (the
# reasoning already rereads it).
SYSTEM_THINK = ("You are a decision engine. Think through the state and the question step by step in your thinking, "
                "weighing the listed options against each other. When your thinking is done, your final reply is the "
                "LETTER of exactly one option and nothing else — no words, no punctuation, no explanation.")
_BARE_KEY = re.compile(r"^([A-Za-z]|\d{1,2}|[A-Za-z]\d)$")
# v3 = v2 + the user message repeated once (as in shisa-de-2: on the second pass a causal model can "see" what comes
# later). Very long messages are not repeated, so context and latency do not double.
REPEAT_SEP = "\n\nRead the same input again before answering:\n\n"
REPEAT_MAX_CHARS = 12000

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


def option_text_v2(name: str, desc: str) -> str:
    """One v2 choice option without its label (see the module docstring)."""
    if not desc:
        return name
    if _BARE_KEY.match(name):
        return desc
    return f"{name}: {desc}"


def build_user_message_v2(d: Decision, state: str | None = None, labels: list[str] | None = None) -> str:
    """The v2 user message of a choice question with at most 26 options."""
    if len(d.options) > MAX_LETTER_OPTIONS:
        raise ValueError(f"{d.id}: v2 layout is for at most {MAX_LETTER_OPTIONS} options")
    labels = labels or LETTERS
    head = [(d.state if state is None else state).rstrip(), "", d.instructions.rstrip()]
    opts = [f"{lab}. {option_text_v2(o.name, o.desc)}" for lab, o in zip(labels, d.options)]
    return "\n".join(head + ["", "Options:"] + opts + ["", "Answer with the letter of exactly one option, and nothing else:"])


def uses_v2_layout(d: Decision) -> bool:
    """Questions that v2 renders differently from v1: choice questions with at most 26 options."""
    return d.type == "choice" and len(d.options) <= MAX_LETTER_OPTIONS


def item_prompt_version(version: str, d: Decision, v1_mix: float, seed: int) -> str:
    """Training only: with prompt v2, a fixed share ``v1_mix`` of the v2-layout questions is rendered in v1, so that the
    trained model also reads v1 (a fallback if v2 turns out worse on some suite category). The choice depends only on
    (seed, question id), so it is the same in every epoch, view and resumed run."""
    if version == "v1" or v1_mix <= 0 or not uses_v2_layout(d):
        return version
    return "v1" if random.Random(f"{seed}/v1mix/{d.id}").random() < v1_mix else version


def chat_messages(d: Decision, state: str | None = None, labels: list[str] | None = None,
                  version: str = "v1", thinking: bool = False) -> list[dict]:
    """The chat messages for one decision in the given prompt version.

    thinking=True (any version) uses the thinking system message, the v2 layout where it applies (v1 otherwise)
    and no repetition: the "letter only" instruction then refers to the reply after the thought.
    """
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version {version!r}; known: {PROMPT_VERSIONS}")
    if thinking:
        body = build_user_message_v2(d, state, labels) if uses_v2_layout(d) else build_user_message(d, state, labels)
        return [{"role": "system", "content": SYSTEM_THINK}, {"role": "user", "content": body}]
    if version == "v3":
        msgs = chat_messages(d, state, labels, "v2")
        user = msgs[-1]["content"]
        if len(user) <= REPEAT_MAX_CHARS:
            msgs[-1] = {"role": "user", "content": user + REPEAT_SEP + user}
        return msgs
    if version == "v1" or not uses_v2_layout(d):
        return [{"role": "user", "content": build_user_message(d, state, labels)}]
    if version == "v2":
        return [{"role": "system", "content": SYSTEM_V2},
                {"role": "user", "content": build_user_message_v2(d, state, labels)}]
    raise ValueError(f"unknown prompt version {version!r}; known: {PROMPT_VERSIONS}")
