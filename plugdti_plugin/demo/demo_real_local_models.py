from __future__ import annotations

"""Real local-model demo.

Edit DRUG_MODEL_DIR and PROTEIN_MODEL_DIR below to your own local pretrained
model directories before running.
"""

from plugdti_plugin_rebuilt.plugdti import ConcatMLPFusionHead, PlugDTIConfig, PlugDTIPlugin
import torch

DRUG_MODEL_DIR = r"E:\预训练的语义模型\chem_bert\chem_bert"
PROTEIN_MODEL_DIR = r"E:\预训练的语义模型\prot_bert_bfd\prot_bert_bfd"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = PlugDTIConfig(
        drug_model_dir=DRUG_MODEL_DIR,
        protein_model_dir=PROTEIN_MODEL_DIR,
        drug_max_length=256,
        protein_max_length=1024,
        common_projection_dim=128,
        freeze_encoders=True,
        device=device,
    )

    plugin = PlugDTIPlugin(config).enable_cache("./embed_cache")

    smiles = ["CCO", "CCN(CC)CC", "CCO"]
    proteins = [
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV",
        "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE",
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV",
    ]

    outputs = plugin(smiles, proteins)
    drug_feat = outputs["drug_embeddings"]
    protein_feat = outputs["protein_embeddings"]
    print("plugin drug embeddings:", tuple(drug_feat.shape))
    print("plugin protein embeddings:", tuple(protein_feat.shape))

    backbone_drug_feat = torch.randn(len(smiles), 64, device=plugin.device)
    backbone_protein_feat = torch.randn(len(smiles), 64, device=plugin.device)

    head = ConcatMLPFusionHead(
        backbone_drug_dim=64,
        backbone_protein_dim=64,
        plugin_drug_dim=plugin.output_drug_dim,
        plugin_protein_dim=plugin.output_protein_dim,
        hidden_dim=128,
        output_dim=1,
    ).to(plugin.device)

    logits = head(backbone_drug_feat, backbone_protein_feat, drug_feat, protein_feat)
    print("downstream logits:", tuple(logits.shape))

    plugin.save_pretrained("./saved_plugdti")
    reloaded = PlugDTIPlugin.from_pretrained("./saved_plugdti", override_device=device)
    reloaded_out = reloaded(smiles[:2], proteins[:2])
    print("reloaded drug shape:", tuple(reloaded_out["drug_embeddings"].shape))


if __name__ == "__main__":
    main()
