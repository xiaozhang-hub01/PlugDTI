from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .cache import EmbeddingDiskCache
from .config import PlugDTIConfig
from .utils import (
    add_protein_spaces,
    apply_pooling,
    choose_device,
    deduplicate_preserve_order,
    ensure_list_str,
)


class PlugDTIPlugin(nn.Module):
    """Extract drug/protein sequence embeddings from local pretrained models.

    The outputs can be concatenated with host DTI/DTA backbone features.
    """

    def __init__(self, config: PlugDTIConfig) -> None:
        super().__init__()
        self.config = config
        self.device_ = choose_device(config.device)

        self.drug_tokenizer = AutoTokenizer.from_pretrained(
            config.drug_model_dir,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        self.protein_tokenizer = AutoTokenizer.from_pretrained(
            config.protein_model_dir,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )

        self.drug_encoder = AutoModel.from_pretrained(
            config.drug_model_dir,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        self.protein_encoder = AutoModel.from_pretrained(
            config.protein_model_dir,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )

        self.drug_hidden_dim = int(self.drug_encoder.config.hidden_size)
        self.protein_hidden_dim = int(self.protein_encoder.config.hidden_size)

        drug_out_dim = config.drug_projection_dim or self.drug_hidden_dim
        protein_out_dim = config.protein_projection_dim or self.protein_hidden_dim

        self.drug_projector: Optional[nn.Module] = None
        self.protein_projector: Optional[nn.Module] = None
        if config.drug_projection_dim is not None:
            self.drug_projector = nn.Sequential(
                nn.Linear(self.drug_hidden_dim, config.drug_projection_dim),
                nn.LayerNorm(config.drug_projection_dim),
            )
        if config.protein_projection_dim is not None:
            self.protein_projector = nn.Sequential(
                nn.Linear(self.protein_hidden_dim, config.protein_projection_dim),
                nn.LayerNorm(config.protein_projection_dim),
            )

        self.output_drug_dim = drug_out_dim
        self.output_protein_dim = protein_out_dim

        self._disk_cache: Optional[EmbeddingDiskCache] = None
        self._mem_cache: Dict[str, Dict[str, torch.Tensor]] = {"drug": {}, "protein": {}}
        self._mem_cache_max = 20000

        self.to(self.device_)
        self._apply_freeze_if_needed()

    @property
    def device(self) -> torch.device:
        return self.device_

    def _apply_freeze_if_needed(self) -> None:
        if self.config.freeze_encoders:
            for p in self.drug_encoder.parameters():
                p.requires_grad = False
            for p in self.protein_encoder.parameters():
                p.requires_grad = False
            self.drug_encoder.eval()
            self.protein_encoder.eval()

    def enable_cache(self, cache_dir: str, mem_cache_max: int = 20000, dtype: Optional[str] = None):
        dtype = dtype or self.config.cache_dtype
        self._disk_cache = EmbeddingDiskCache(cache_dir=cache_dir, dtype=dtype)
        self._mem_cache_max = int(mem_cache_max)
        return self

    def disable_cache(self):
        self._disk_cache = None
        return self

    def clear_memory_cache(self):
        self._mem_cache = {"drug": {}, "protein": {}}
        return self

    def _cache_prefix(self, namespace: str) -> str:
        if namespace == "drug":
            model_dir = self.config.drug_model_dir
            pooling = self.config.drug_pooling
            max_len = self.config.drug_max_length
            proj_dim = self.config.drug_projection_dim
        elif namespace == "protein":
            model_dir = self.config.protein_model_dir
            pooling = self.config.protein_pooling
            max_len = self.config.protein_max_length
            proj_dim = self.config.protein_projection_dim
        else:
            raise ValueError(f"Unsupported namespace: {namespace}")
        return f"{namespace}|model={model_dir}|pool={pooling}|max_len={max_len}|proj={proj_dim}"

    def _prune_memory_cache(self, namespace: str) -> None:
        mem = self._mem_cache[namespace]
        if len(mem) <= self._mem_cache_max:
            return
        keys = list(mem.keys())
        for key in keys[: len(keys) // 2]:
            mem.pop(key, None)

    def _tokenize_drugs(self, smiles_list: Sequence[str]) -> Dict[str, torch.Tensor]:
        return self.drug_tokenizer(
            list(smiles_list),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.drug_max_length,
        )

    def _tokenize_proteins(self, seq_list: Sequence[str]) -> Dict[str, torch.Tensor]:
        processed = list(seq_list)
        if self.config.protein_add_spaces:
            processed = [add_protein_spaces(seq) for seq in processed]
        return self.protein_tokenizer(
            processed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.protein_max_length,
        )

    def _encode_batch(self, namespace: str, texts: Sequence[str]) -> torch.Tensor:
        if namespace == "drug":
            batch = self._tokenize_drugs(texts)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.drug_encoder(**batch)
            pooled = apply_pooling(outputs.last_hidden_state, batch["attention_mask"], self.config.drug_pooling)
            if self.drug_projector is not None:
                pooled = self.drug_projector(pooled)
            return pooled

        if namespace == "protein":
            batch = self._tokenize_proteins(texts)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.protein_encoder(**batch)
            pooled = apply_pooling(outputs.last_hidden_state, batch["attention_mask"], self.config.protein_pooling)
            if self.protein_projector is not None:
                pooled = self.protein_projector(pooled)
            return pooled

        raise ValueError(f"Unsupported namespace: {namespace}")

    def _encode_with_cache(self, namespace: str, texts: Sequence[str]) -> torch.Tensor:
        prefix = self._cache_prefix(namespace)
        unique_texts, back_indices = deduplicate_preserve_order(texts)
        mem = self._mem_cache[namespace]
        cached_vectors: List[Optional[torch.Tensor]] = [None] * len(unique_texts)
        misses: List[str] = []
        miss_positions: List[int] = []

        for i, text in enumerate(unique_texts):
            key = prefix + "\n" + text
            if key in mem:
                cached_vectors[i] = mem[key]
                continue
            if self._disk_cache is not None:
                arr = self._disk_cache.get(namespace, key)
                if arr is not None:
                    ten = torch.from_numpy(arr).to(self.device)
                    cached_vectors[i] = ten
                    mem[key] = ten
                    continue
            misses.append(text)
            miss_positions.append(i)

        if misses:
            use_grad = not self.config.freeze_encoders
            context = torch.enable_grad() if use_grad else torch.inference_mode()
            with context:
                miss_emb = self._encode_batch(namespace, misses)
            miss_emb = miss_emb.detach()
            for local_idx, unique_idx in enumerate(miss_positions):
                text = unique_texts[unique_idx]
                key = prefix + "\n" + text
                vec = miss_emb[local_idx]
                cached_vectors[unique_idx] = vec
                mem[key] = vec
                if self._disk_cache is not None:
                    self._disk_cache.set(namespace, key, vec.cpu().numpy())
            self._prune_memory_cache(namespace)

        stacked = torch.stack([cached_vectors[i] for i in back_indices], dim=0)
        return stacked

    def encode_drugs(self, smiles_list: Sequence[str], use_cache: bool = True) -> torch.Tensor:
        smiles_list = ensure_list_str(smiles_list, "smiles_list")
        if use_cache and self._disk_cache is not None:
            return self._encode_with_cache("drug", smiles_list)
        use_grad = not self.config.freeze_encoders
        context = torch.enable_grad() if use_grad else torch.inference_mode()
        with context:
            return self._encode_batch("drug", smiles_list)

    def encode_proteins(self, protein_list: Sequence[str], use_cache: bool = True) -> torch.Tensor:
        protein_list = ensure_list_str(protein_list, "protein_list")
        if use_cache and self._disk_cache is not None:
            return self._encode_with_cache("protein", protein_list)
        use_grad = not self.config.freeze_encoders
        context = torch.enable_grad() if use_grad else torch.inference_mode()
        with context:
            return self._encode_batch("protein", protein_list)

    def forward(self, smiles_list: Sequence[str], protein_list: Sequence[str], use_cache: bool = True):
        drug_embeddings = self.encode_drugs(smiles_list, use_cache=use_cache)
        protein_embeddings = self.encode_proteins(protein_list, use_cache=use_cache)
        return {
            "drug_embeddings": drug_embeddings,
            "protein_embeddings": protein_embeddings,
        }

    def save_pretrained(self, save_dir: str, save_backbone_weights: bool = False) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_json(save_dir / "plugin_config.json")

        payload = {
            "drug_projector": self.drug_projector.state_dict() if self.drug_projector is not None else None,
            "protein_projector": self.protein_projector.state_dict() if self.protein_projector is not None else None,
        }
        torch.save(payload, save_dir / "plugin_state.pt")

        if save_backbone_weights:
            self.drug_encoder.save_pretrained(save_dir / "drug_encoder")
            self.protein_encoder.save_pretrained(save_dir / "protein_encoder")
            self.drug_tokenizer.save_pretrained(save_dir / "drug_encoder")
            self.protein_tokenizer.save_pretrained(save_dir / "protein_encoder")

    @classmethod
    def from_pretrained(cls, save_dir: str, override_device: Optional[str] = None) -> "PlugDTIPlugin":
        save_dir = Path(save_dir)
        config = PlugDTIConfig.load_json(save_dir / "plugin_config.json")
        if override_device is not None:
            config.device = override_device
        model = cls(config)
        payload = torch.load(save_dir / "plugin_state.pt", map_location=model.device)
        if model.drug_projector is not None and payload["drug_projector"] is not None:
            model.drug_projector.load_state_dict(payload["drug_projector"])
        if model.protein_projector is not None and payload["protein_projector"] is not None:
            model.protein_projector.load_state_dict(payload["protein_projector"])
        return model
