"""Option labels: A-Z, then two-letter codes that are single tokens (no torch needed)."""

from __future__ import annotations

from ajev.lm.prompt import LETTERS, MAX_OPTIONS

_LABEL_CACHE: dict[int, tuple[list[str], list[list[int]], list[list[bool]]]] = {}


def option_labels(tok) -> tuple[list[str], list[list[int]], list[list[bool]]]:
    """Up to 255 labels: A-Z, then AA, AB, ... that the tokenizer keeps as one token.

    Returns (labels, token ids [N, 2], valid [N, 2]): each label has up to two single-token forms
    ("AB" and " AB"); a missing second form repeats the first and is marked invalid.
    A two-letter code must also stay one token when appended after the chat template's answer position.
    Cached per tokenizer.
    """
    key = id(tok)
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]
    ctx = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False)
    ctx_ids = tok.encode(ctx, add_special_tokens=False)

    def forms_of(label: str) -> list[int]:
        forms = []
        for form in (label, " " + label):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1 and t[0] not in forms:
                forms.append(t[0])
        return forms

    labels, ids, valid, seen = [], [], [], set()
    candidates = list(LETTERS) + [a + b for a in LETTERS for b in LETTERS]
    for label in candidates:
        if len(labels) == MAX_OPTIONS:
            break
        forms = forms_of(label)
        if len(label) == 1 and not forms:
            raise ValueError(f"letter {label!r} is not a single token for this tokenizer")
        if not forms or forms[0] in seen:
            continue
        if len(label) > 1 and tok.encode(ctx + label, add_special_tokens=False) != ctx_ids + forms[:1]:
            continue  # not a single token at the answer position
        seen.update(forms)
        labels.append(label)
        ids.append(forms + [forms[0]] * (2 - len(forms)))
        valid.append([True] * len(forms) + [False] * (2 - len(forms)))
    _LABEL_CACHE[key] = (labels, ids, valid)
    return _LABEL_CACHE[key]
