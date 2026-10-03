"""LLM predictor (Gemma 4 12B, zero-shot or with a LoRA adapter).

    from ajev.lm.predictor import LMPredictor
    probs = LMPredictor("google/gemma-4-12B-it", adapter="andyzhang232/ajev-gemma4-12b-lora5").predict(decisions)

Each decision is rendered with ajev/lm/prompt.py and the chat template. One forward pass reads the
last-position logits of the option labels, which are softmaxed after dividing by a per-type temperature.
``prompt_ids``, ``left_pad`` and ``letter_logits`` are shared with training, so training and inference read
exactly the same positions.
"""

from __future__ import annotations

import json
import os

import torch

from ajev.lm.labels import option_labels
from ajev.lm.prompt import MAX_LETTER_OPTIONS, build_user_message
from ajev.schema import Decision

# Adapter-side config: base model, training state limit, step, calibration temperatures.
LM_CONFIG = "ajev_lm_config.json"
# Default state limit (tokens) at inference.
DEFAULT_INFER_STATE_TOKENS = 16384


def default_device() -> str:
    """cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_model(model_id: str, dtype: torch.dtype = torch.bfloat16):
    """Gemma 4 may need the image-text-to-text class; only text is fed."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except (ValueError, KeyError):
        return AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    return tok


def letter_token_table(tok) -> tuple[torch.Tensor, torch.Tensor]:
    """Label token ids [N, 2] and validity mask [N, 2] (see ``option_labels``)."""
    _, ids, valid = option_labels(tok)
    return torch.tensor(ids), torch.tensor(valid)


def prompt_ids(tok, d: Decision, max_state_tokens: int) -> list[int]:
    """Token ids for one decision, ending at the answer position. Longer states are truncated."""
    state_ids = tok.encode(d.state, add_special_tokens=False)
    state = d.state if len(state_ids) <= max_state_tokens else \
        tok.decode(state_ids[:max_state_tokens]) + " …[truncated]"
    labels = option_labels(tok)[0] if len(d.options) > MAX_LETTER_OPTIONS else None
    messages = [{"role": "user", "content": build_user_message(d, state, labels)}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    # Some transformers versions return a dict.
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    return list(ids)


def left_pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad so that every sequence ends at the last column."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, width - len(s):] = torch.tensor(s)
        mask[r, width - len(s):] = 1
    return ids, mask


def letter_logits(model, ids: torch.Tensor, mask: torch.Tensor, table: torch.Tensor, valid: torch.Tensor,
                  k: int) -> torch.Tensor:
    """Scores [B, k] of the first k labels at the last position (float32)."""
    try:
        out = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1].float()
    except TypeError:  # no logits_to_keep in older versions
        out = model(input_ids=ids, attention_mask=mask).logits[:, -1].float()
    return label_scores(out, table, valid, k)


def label_scores(out: torch.Tensor, table: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """Vocabulary logits [B, V] -> label scores [B, k]; the two forms of a label are logsumexp-ed."""
    table, valid = table.to(out.device), valid.to(out.device)
    per_form = out[:, table].masked_fill(~valid, float("-inf"))  # [B, N, 2]
    return torch.logsumexp(per_form, dim=-1)[:, :k]


def common_prefix_len(seqs: list[list[int]]) -> int:
    """Length of the longest common token prefix."""
    n = min(len(s) for s in seqs)
    first = seqs[0]
    for j in range(n):
        if any(s[j] != first[j] for s in seqs[1:]):
            return j
    return n


def resolve_adapter(adapter: str | None) -> str | None:
    """Local adapter directory, or a Hub repo id downloaded on first use."""
    if not adapter or os.path.isdir(adapter):
        return adapter
    from huggingface_hub import snapshot_download

    return snapshot_download(adapter, allow_patterns=["adapter_config.json", "adapter_model.safetensors", LM_CONFIG])


def read_lm_config(adapter: str | None) -> dict:
    p = os.path.join(adapter, LM_CONFIG) if adapter else ""
    if not adapter or not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


class LMPredictor:
    """Zero-shot or LoRA predictor.

    Args:
        model_id: base model, e.g. "google/gemma-4-12B-it".
        adapter: LoRA adapter directory or Hub repo id; None for zero-shot.
        temperatures: per-type temperatures; None reads them from the adapter, {} disables them.
        merge: merge the LoRA in memory (faster). Do not save and reload a merged Gemma 4 checkpoint.
        prefix_cache: score questions that share a state on one cached prefix.
    """

    def __init__(self, model_id: str, adapter: str | None = None, device: str | None = None,
                 batch_tokens: int = 24000, max_state_tokens: int | None = None,
                 dtype: torch.dtype = torch.bfloat16, temperatures: dict[str, float] | None = None,
                 model=None, tok=None, merge: bool = False, prefix_cache: bool = False) -> None:
        adapter = resolve_adapter(adapter)
        cfg = read_lm_config(adapter)
        self.tok = tok or load_tokenizer(model_id)
        self.device = torch.device(device or default_device())
        if model is None:
            model = load_base_model(model_id, dtype)
            if adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adapter)
                if merge:
                    model = model.merge_and_unload()
            model = model.to(self.device)
        self.model = model.eval()
        self.batch_tokens = batch_tokens
        self.max_state_tokens = max_state_tokens or DEFAULT_INFER_STATE_TOKENS
        self.temperatures = temperatures if temperatures is not None else cfg.get("temperatures", {})
        self.table, self.valid = letter_token_table(self.tok)
        # Over 26 options: "codes" reads two-letter codes; "knockout" uses groups of 26. Beyond the label table
        # knockout is always used.
        self.wide_mode = "codes"
        self.knockout_per_group = 2
        self.prefix_cache = prefix_cache
        self.min_shared_prefix = 64
        self.prefix_cache_bytes = 16 * 2 ** 30  # memory budget for copies of the prefix cache

    @torch.no_grad()
    def predict_logits(self, decisions: list[Decision]) -> list[list[float]]:
        """Per-option logits for each decision."""
        limit = MAX_LETTER_OPTIONS if self.wide_mode == "knockout" else len(option_labels(self.tok)[0])
        narrow = [i for i, d in enumerate(decisions) if len(d.options) <= limit]
        wide = [i for i, d in enumerate(decisions) if len(d.options) > limit]
        out: list[list[float]] = [[0.0] * len(d.options) for d in decisions]
        for i, lg in zip(narrow, self._letter_scores([decisions[i] for i in narrow])):
            out[i] = lg
        if wide:
            for i, lg in zip(wide, self._knockout([decisions[i] for i in wide])):
                out[i] = lg
        return out

    def _knockout(self, decisions: list[Decision]) -> list[list[float]]:
        """Score groups of 26, then a final round of the top options per group (recursive)."""
        from ajev.lm.knockout import combine, finalists, split_groups

        plans, subs = [], []
        for d in decisions:
            groups = split_groups(len(d.options), MAX_LETTER_OPTIONS)
            plans.append((d, groups, len(subs)))
            for gi, g in enumerate(groups):
                subs.append(Decision(**{**d.__dict__, "id": f"{d.id}#g{gi}", "options": [d.options[o] for o in g],
                                        "target": [1.0 / len(g)] * len(g)}))
        group_scores = self._letter_scores(subs)
        finals, idxs = [], []
        for d, groups, s0 in plans:
            f = finalists(groups, group_scores[s0: s0 + len(groups)], self.knockout_per_group)
            idxs.append(f)
            finals.append(Decision(**{**d.__dict__, "id": f"{d.id}#final", "options": [d.options[o] for o in f],
                                      "target": [1.0 / len(f)] * len(f)}))
        final_scores = self.predict_logits(finals)
        return [combine(len(d.options), groups, group_scores[s0: s0 + len(groups)], f, fs)
                for (d, groups, s0), f, fs in zip(plans, idxs, final_scores)]

    @torch.no_grad()
    def _letter_scores(self, decisions: list[Decision]) -> list[list[float]]:
        """One forward pass per decision, batched by token budget."""
        was_training = self.model.training
        self.model.eval()
        out: list[list[float]] = [[0.0] * len(d.options) for d in decisions]
        todo = [(i, prompt_ids(self.tok, d, self.max_state_tokens)) for i, d in enumerate(decisions)]
        if self.prefix_cache:
            todo = self._shared_prefix_scores(decisions, todo, out)
        # Sort by length to minimise padding.
        todo.sort(key=lambda x: len(x[1]))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        batch: list[tuple[int, list[int]]] = []

        def flush() -> None:
            if not batch:
                return
            ids, mask = left_pad([s for _, s in batch], pad)
            k = max(len(decisions[i].options) for i, _ in batch)
            scores = letter_logits(self.model, ids.to(self.device), mask.to(self.device), self.table, self.valid, k)
            for r, (i, _) in enumerate(batch):
                out[i] = scores[r, : len(decisions[i].options)].tolist()
            batch.clear()

        for item in todo:
            if batch and (len(batch) + 1) * len(item[1]) > self.batch_tokens:
                flush()
            batch.append(item)
        flush()
        if was_training:
            self.model.train()
        return out

    def _shared_prefix_scores(self, decisions: list[Decision], todo: list[tuple[int, list[int]]],
                              out: list[list[float]]) -> list[tuple[int, list[int]]]:
        """Score decisions that share a state on one prefix KV cache; return the ones left over.

        The prefix is the longest common prefix of the full prompt token ids (not of the state alone, which
        can tokenize differently at the boundary). Tails are right-padded and run on copies of the cache.
        """
        groups: dict[str, list[tuple[int, list[int]]]] = {}
        for item in todo:
            groups.setdefault(decisions[item[0]].state, []).append(item)
        rest = []
        for items in groups.values():
            p = common_prefix_len([s for _, s in items]) if len(items) > 1 else 0
            p = min(p, min(len(s) for _, s in items) - 1)  # keep at least one tail token
            if p < self.min_shared_prefix:
                rest.extend(items)
                continue
            self._score_with_prefix(decisions, items, p, out)
        return rest

    def _score_with_prefix(self, decisions: list[Decision], items: list[tuple[int, list[int]]], p: int,
                           out: list[list[float]]) -> None:
        import copy

        prefix = torch.tensor([items[0][1][:p]], device=self.device)
        cache = self.model(input_ids=prefix, use_cache=True, logits_to_keep=1).past_key_values
        # Size of one copy of the prefix cache, to bound the tails per batch.
        row_bytes = sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        tails = sorted(((i, s[p:]) for i, s in items), key=lambda x: len(x[1]))
        start = 0
        while start < len(tails):
            end = start + 1
            while end < len(tails) and (end - start + 1) * len(tails[end][1]) <= self.batch_tokens \
                    and (end - start + 1) * row_bytes <= self.prefix_cache_bytes:
                end += 1
            chunk = tails[start:end]
            start = end
            width = max(len(t) for _, t in chunk)
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), p + width), dtype=torch.long)
            mask[:, :p] = 1
            for r, (_, t) in enumerate(chunk):  # right padding keeps positions identical
                ids[r, : len(t)] = torch.tensor(t)
                mask[r, p: p + len(t)] = 1
            c = copy.deepcopy(cache)
            c.batch_repeat_interleave(len(chunk))
            last = [len(t) - 1 for _, t in chunk]
            keep = sorted(set(last))
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                past_key_values=c, use_cache=True,
                                logits_to_keep=torch.tensor(keep, device=self.device)).logits
            rows = logits[torch.arange(len(chunk)), torch.tensor([keep.index(j) for j in last])].float()
            k = max(len(decisions[i].options) for i, _ in chunk)
            scores = label_scores(rows, self.table, self.valid, k)
            for r, (i, _) in enumerate(chunk):
                out[i] = scores[r, : len(decisions[i].options)].tolist()
            del c, logits

    def predict(self, decisions: list[Decision]) -> list[list[float]]:
        """Option probabilities: softmax(logits / temperature of the question type)."""
        probs = []
        for d, lg in zip(decisions, self.predict_logits(decisions)):
            t = torch.tensor(lg) / self.temperatures.get(d.type, 1.0)
            probs.append(torch.softmax(t, dim=-1).tolist())
        return probs
