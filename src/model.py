from __future__ import annotations

import torch
from torch import nn

from src import config


class GRUFraudModel(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        n_categories: int,
        emb_dim: int = config.CAT_EMB_DIM,
        proj_dim: int = config.NUM_PROJ_DIM,
        hidden: int = config.GRU_HIDDEN,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.cat_emb = nn.Embedding(n_categories + 1, emb_dim, padding_idx=0)
        self.num_proj = nn.Linear(n_numeric, proj_dim)
        self.gru = nn.GRU(
            input_size=proj_dim + emb_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb = self.cat_emb(x_cat)
        proj = torch.relu(self.num_proj(x_num))
        seq = torch.cat([proj, emb], dim=-1)
        out, _ = self.gru(seq)
        last = out[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
