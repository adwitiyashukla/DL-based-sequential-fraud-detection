"""
Per card sequence windows, built on demand.

The naive approach is to materialise an array of shape
(1.29M sequences, 10 timesteps, 11 features), which duplicates the dataset
tenfold and costs several gigabytes for no benefit. Instead this module keeps
one flat float32 array per split and slices the window inside __getitem__.
The only extra state needed is `block_start`, which records where each card's
block of rows begins so that a window can never run off the start of a card and
pick up the tail of a different cardholder's history.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src import config


def load_cache() -> dict:
    """Load the arrays written by src/features.py."""
    if not config.FEATURE_CACHE.exists():
        raise FileNotFoundError(
            f"{config.FEATURE_CACHE} not found. Run 'python -m src.features' first."
        )
    with np.load(config.FEATURE_CACHE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def split_indices(is_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Global row indices for each split, both in ascending order.

    Ascending order is the canonical test ordering. Every model writes its test
    scores in this order so that evaluate.py can line them up against the
    labels and the transaction amounts without any further bookkeeping.
    """
    train_rows = np.flatnonzero(~is_test).astype(np.int64)
    test_rows = np.flatnonzero(is_test).astype(np.int64)
    return train_rows, test_rows


def subsample_negatives(
    rows: np.ndarray, y: np.ndarray, keep_fraction: float, seed: int = config.SEED
) -> np.ndarray:
    """
    Keep every fraud row and a random fraction of the legitimate rows.

    This is a training-time speed lever only. The test set is never subsampled,
    because the metrics are only meaningful at the true base rate.
    """
    rng = np.random.default_rng(seed)
    labels = y[rows]
    positives = rows[labels == 1]
    negatives = rows[labels == 0]
    keep_n = int(round(len(negatives) * keep_fraction))
    kept = rng.choice(negatives, size=keep_n, replace=False)
    out = np.sort(np.concatenate([positives, kept]))
    return out.astype(np.int64)


class SequenceDataset(Dataset):
    """
    Yields the last `seq_len` transactions of a card, ending at the target row.

    Returns
    -------
    x_num : float32 tensor of shape (seq_len, n_features + 1)
        The final column is a validity flag: 1.0 for a real transaction and
        0.0 for a padded timestep. Without it the model cannot tell a padded
        step from a real transaction whose standardised features happen to sit
        at the mean.
    x_cat : int64 tensor of shape (seq_len,)
        Category index. 0 is reserved for padding.
    y : float32 scalar
        The label of the target row only. The earlier timesteps supply context,
        not supervision.
    """

    def __init__(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        y: np.ndarray,
        block_start: np.ndarray,
        rows: np.ndarray,
        seq_len: int = config.SEQ_LEN,
    ) -> None:
        self.x_num = x_num
        self.x_cat = x_cat
        self.y = y
        self.block_start = block_start
        self.rows = rows
        self.seq_len = seq_len
        self.n_features = x_num.shape[1]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = int(self.rows[i])

        # Walk backwards at most seq_len - 1 steps, but never past the first
        # transaction of this card.
        start = max(int(self.block_start[row]), row - self.seq_len + 1)
        n_real = row - start + 1

        num = np.zeros((self.seq_len, self.n_features + 1), dtype=np.float32)
        cat = np.zeros(self.seq_len, dtype=np.int64)

        # Left padding: the real history sits at the end of the window, so the
        # final timestep is always the transaction being scored.
        num[-n_real:, : self.n_features] = self.x_num[start : row + 1]
        num[-n_real:, self.n_features] = 1.0
        cat[-n_real:] = self.x_cat[start : row + 1]

        return torch.from_numpy(num), torch.from_numpy(cat), torch.tensor(
            self.y[row], dtype=torch.float32
        )
