from __future__ import annotations

"""Self-contained smoke demo.
It first builds tiny local BERT-like encoders and tokenizers, then runs PlugDTI.
This demo does not need internet access.
"""

from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from plugdti_plugin.plugdti import ConcatMLPFusionHead, PlugDTIConfig, PlugDTIPlugin


def build_tiny_local_encoder(save_dir: str, vocab_tokens: list[str], hidden_size: int) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = save_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(save_dir)

    config = BertConfig(
        vocab_size=len(vocab_tokens),
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=2048,
        type_vocab_size=2,
        pad_token_id=0,
    )
    model = BertModel(config)
    model.save_pretrained(save_dir)


def main() -> None:
    root = Path("./tiny_local_models")
    drug_dir = root / "tiny_drug_encoder"
    prot_dir = root / "tiny_protein_encoder"

    base_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    smiles_tokens = base_tokens + list("CONSPFBrcnosp[]=#()123456789+-/@") + [chr(92)]
    protein_tokens = base_tokens + list("ACDEFGHIKLMNPQRSTVWYXBZUO")

    if not drug_dir.exists():
        build_tiny_local_encoder(str(drug_dir), smiles_tokens, hidden_size=48)
    if not prot_dir.exists():
        build_tiny_local_encoder(str(prot_dir), protein_tokens, hidden_size=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PlugDTIConfig(
        drug_model_dir=str(drug_dir),
        protein_model_dir=str(prot_dir),
        drug_max_length=64,
        protein_max_length=128,
        common_projection_dim=32,
        freeze_encoders=True,
        device=device,
    )
    plugin = PlugDTIPlugin(config).enable_cache("./tiny_cache")

    smiles = ["CCO", "CCN(CC)CC", "C1=CC=CC=C1"]
    proteins = [
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY",
        "MKWVTFISLLLLFSSAYSRGVFRR",
        "GAVLIPFWMCSTYNQDEKRH",
    ]

    outputs = plugin(smiles, proteins)
    drug_feat = outputs["drug_embeddings"]
    protein_feat = outputs["protein_embeddings"]

    print("drug embedding shape:", tuple(drug_feat.shape))
    print("protein embedding shape:", tuple(protein_feat.shape))

    backbone_drug_feat = torch.randn(len(smiles), 16, device=plugin.device)
    backbone_protein_feat = torch.randn(len(smiles), 24, device=plugin.device)
    fusion = ConcatMLPFusionHead(
        backbone_drug_dim=16,
        backbone_protein_dim=24,
        plugin_drug_dim=plugin.output_drug_dim,
        plugin_protein_dim=plugin.output_protein_dim,
        hidden_dim=64,
        output_dim=1,
    ).to(plugin.device)
    logits = fusion(backbone_drug_feat, backbone_protein_feat, drug_feat, protein_feat)
    print("downstream logits shape:", tuple(logits.shape))

    plugin.save_pretrained("./tiny_saved_plugin")
    loaded = PlugDTIPlugin.from_pretrained("./tiny_saved_plugin", override_device=device)
    loaded_out = loaded(smiles[:2], proteins[:2])
    print("reloaded protein shape:", tuple(loaded_out["protein_embeddings"].shape))


if __name__ == "__main__":
    main()
