from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Sequence

import torch


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ensure_list_str(items: Sequence[str], name: str) -> List[str]:
    if isinstance(items, str):
        raise TypeError(f"{name} must be a sequence of strings, not a single string")
    out = []
    for idx, item in enumerate(items):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{idx}] must be a string, got {type(item)!r}")
        item = item.strip()
        if not item:
            raise ValueError(f"{name}[{idx}] is empty")
        out.append(item)
    return out


def choose_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_protein_spaces(sequence: str) -> str:
    return " ".join(list(sequence.strip()))


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def masked_max_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).bool()
    masked = last_hidden_state.masked_fill(~mask, torch.finfo(last_hidden_state.dtype).min)
    return masked.max(dim=1).values


def apply_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    if pooling == "mean":
        return masked_mean_pool(last_hidden_state, attention_mask)
    if pooling == "max":
        return masked_max_pool(last_hidden_state, attention_mask)
    raise ValueError(f"Unsupported pooling: {pooling}")


def deduplicate_preserve_order(items: Sequence[str]):
    unique_items = []
    item_to_idx = {}
    back_indices = []
    for item in items:
        if item in item_to_idx:
            back_indices.append(item_to_idx[item])
        else:
            item_to_idx[item] = len(unique_items)
            unique_items.append(item)
            back_indices.append(item_to_idx[item])
    return unique_items, back_indices


def mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
