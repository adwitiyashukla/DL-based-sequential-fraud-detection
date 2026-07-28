"""
The sequence model.

A single layer GRU over the card's recent transaction history. The design is
deliberately small: the point of the project is to isolate the value of
sequence context, not to win a capacity contest against the gradient boosting
baseline. A larger network would confound the two.
"""

from __future__ import annotations

import torch
from torch import nn

from src import config


class GRUFraudModel(nn.Module):
    """
    Per timestep the model sees:
      - a learned embedding of the merchant category
      - a linear projection of the standardised numeric features

    Those are concatenated and fed to the GRU. The hidden state at the final
    timestep, which corresponds to the transaction being scored, goes through
    dropout and a linear head to a single logit.

    The model outputs a raw logit rather than a probability because training
    uses BCEWithLogitsLoss, which folds the sigmoid into the loss for numerical
    stability.
    """

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
        # padding_idx=0 keeps the padding embedding pinned at zero and excluded
        # from gradient updates.
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
        emb = self.cat_emb(x_cat)                      # (B, T, emb_dim)
        proj = torch.relu(self.num_proj(x_num))        # (B, T, proj_dim)
        seq = torch.cat([proj, emb], dim=-1)           # (B, T, emb + proj)
        out, _ = self.gru(seq)                         # (B, T, hidden)
        last = out[:, -1, :]                           # (B, hidden)
        return self.head(self.dropout(last)).squeeze(-1)   # (B,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
