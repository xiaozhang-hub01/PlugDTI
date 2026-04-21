from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .utils import mkdir, sha1_text


class EmbeddingDiskCache:
    """Simple hashed NPY cache for embeddings."""

    def __init__(self, cache_dir: str | Path, dtype: str = "float16") -> None:
        if dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be 'float16' or 'float32'")
        self.cache_dir = mkdir(cache_dir)
        self.dtype = np.float16 if dtype == "float16" else np.float32

    def _ns_dir(self, namespace: str) -> Path:
        return mkdir(self.cache_dir / namespace)

    def _file_path(self, namespace: str, key: str) -> Path:
        return self._ns_dir(namespace) / f"{sha1_text(key)}.npy"

    def _meta_path(self, namespace: str, key: str) -> Path:
        return self._ns_dir(namespace) / f"{sha1_text(key)}.json"

    def get(self, namespace: str, key: str) -> Optional[np.ndarray]:
        file_path = self._file_path(namespace, key)
        if not file_path.exists():
            return None
        arr = np.load(file_path)
        if arr.dtype != self.dtype:
            arr = arr.astype(self.dtype, copy=False)
        return arr

    def set(self, namespace: str, key: str, value: np.ndarray) -> None:
        file_path = self._file_path(namespace, key)
        meta_path = self._meta_path(namespace, key)
        arr = np.asarray(value, dtype=self.dtype)
        np.save(file_path, arr)
        meta_path.write_text(json.dumps({"namespace": namespace, "shape": list(arr.shape)}), encoding="utf-8")
