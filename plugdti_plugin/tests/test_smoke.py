from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.demo_tiny_local_models import build_tiny_local_encoder  # noqa: E402
from plugdti import PlugDTIConfig, PlugDTIPlugin  # noqa: E402


def main() -> None:
    work = ROOT / "tests" / "tmp_test_assets"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    drug_dir = work / "drug"
    prot_dir = work / "prot"

    base_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    smiles_vocab = base_tokens + list("CONSPFBrcnosp[]=#()123456789+-/@") + [chr(92)]
    protein_vocab = base_tokens + list("ACDEFGHIKLMNPQRSTVWYXBZUO")
    build_tiny_local_encoder(str(drug_dir), smiles_vocab, hidden_size=32)
    build_tiny_local_encoder(str(prot_dir), protein_vocab, hidden_size=40)

    config = PlugDTIConfig(
        drug_model_dir=str(drug_dir),
        protein_model_dir=str(prot_dir),
        common_projection_dim=24,
        drug_max_length=64,
        protein_max_length=128,
        freeze_encoders=True,
        device="cpu",
    )
    plugin = PlugDTIPlugin(config).enable_cache(str(work / "cache"))
    out = plugin(["CCO", "CCO"], ["MTEYKLVV", "MTEYKLVV"])
    assert tuple(out["drug_embeddings"].shape) == (2, 24)
    assert tuple(out["protein_embeddings"].shape) == (2, 24)

    plugin.save_pretrained(str(work / "saved"))
    loaded = PlugDTIPlugin.from_pretrained(str(work / "saved"), override_device="cpu")
    out2 = loaded(["CCN"], ["MKWVTFIS"])
    assert tuple(out2["drug_embeddings"].shape) == (1, 24)
    assert tuple(out2["protein_embeddings"].shape) == (1, 24)
    print("smoke test passed")


if __name__ == "__main__":
    main()
