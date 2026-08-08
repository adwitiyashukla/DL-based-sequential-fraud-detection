"""
The GRU, copied verbatim from the training repository so the Space is
self-contained. Architecture must match the checkpoint exactly.

Source: https://github.com/adwitiyashukla/DL-based-sequential-fraud-detection
"""

from __future__ import annotations

import torch
from torch import nn


class GRUFraudModel(nn.Module):
    """
    Per timestep the model sees a learned embedding of the merchant category
    and a linear projection of the standardised numeric features. Those are
    concatenated and fed to a single layer GRU. The hidden state at the final
    timestep, which is the transaction being scored, goes through dropout and a
    linear head to one logit.
    """

    def __init__(
        self,
        n_numeric: int,
        n_categories: int,
        emb_dim: int = 16,
        proj_dim: int = 32,
        hidden: int = 64,
        dropout: float = 0.2,
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
        out, _ = self.gru(torch.cat([proj, emb], dim=-1))
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)
