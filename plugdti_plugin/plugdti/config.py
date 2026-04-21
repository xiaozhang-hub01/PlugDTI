from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PlugDTIConfig:
    drug_model_dir: str
    protein_model_dir: str
    drug_max_length: int = 256
    protein_max_length: int = 1024
    drug_pooling: str = "mean"          # mean | cls | max
    protein_pooling: str = "mean"       # mean | cls | max
    drug_projection_dim: Optional[int] = None
    protein_projection_dim: Optional[int] = None
    common_projection_dim: Optional[int] = None
    freeze_encoders: bool = True
    protein_add_spaces: bool = True
    local_files_only: bool = True
    trust_remote_code: bool = False
    cache_dtype: str = "float16"        # float16 | float32
    device: Optional[str] = None

    def __post_init__(self) -> None:
        valid_poolings = {"mean", "cls", "max"}
        if self.drug_pooling not in valid_poolings:
            raise ValueError(f"drug_pooling must be one of {valid_poolings}, got {self.drug_pooling}")
        if self.protein_pooling not in valid_poolings:
            raise ValueError(f"protein_pooling must be one of {valid_poolings}, got {self.protein_pooling}")
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache_dtype must be 'float16' or 'float32'")

        if self.common_projection_dim is not None:
            self.drug_projection_dim = self.common_projection_dim
            self.protein_projection_dim = self.common_projection_dim

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlugDTIConfig":
        return cls(**data)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "PlugDTIConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
