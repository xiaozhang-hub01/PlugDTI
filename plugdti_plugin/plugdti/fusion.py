from __future__ import annotations

import torch
import torch.nn as nn


class ConcatMLPFusionHead(nn.Module):
    """A small downstream head showing how to concatenate plugin embeddings
    with original drug/protein backbone features.
    """

    def __init__(
        self,
        backbone_drug_dim: int,
        backbone_protein_dim: int,
        plugin_drug_dim: int,
        plugin_protein_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = backbone_drug_dim + backbone_protein_dim + plugin_drug_dim + plugin_protein_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        backbone_drug_feat: torch.Tensor,
        backbone_protein_feat: torch.Tensor,
        plugin_drug_feat: torch.Tensor,
        plugin_protein_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [backbone_drug_feat, backbone_protein_feat, plugin_drug_feat, plugin_protein_feat],
            dim=-1,
        )
        return self.mlp(x)
