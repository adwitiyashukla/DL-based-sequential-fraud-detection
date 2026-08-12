from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src import config


def load_cache() -> dict:
    if not config.FEATURE_CACHE.exists():
        raise FileNotFoundError(
            f"{config.FEATURE_CACHE} not found. Run 'python -m src.features' first."
        )
    with np.load(config.FEATURE_CACHE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def split_indices(is_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_rows = np.flatnonzero(~is_test).astype(np.int64)
    test_rows = np.flatnonzero(is_test).astype(np.int64)
    return train_rows, test_rows


def subsample_negatives(
    rows: np.ndarray, y: np.ndarray, keep_fraction: float, seed: int = config.SEED
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = y[rows]
    positives = rows[labels == 1]
    negatives = rows[labels == 0]
    keep_n = int(round(len(negatives) * keep_fraction))
    kept = rng.choice(negatives, size=keep_n, replace=False)
    out = np.sort(np.concatenate([positives, kept]))
    return out.astype(np.int64)


class SequenceDataset(Dataset):
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

        start = max(int(self.block_start[row]), row - self.seq_len + 1)
        n_real = row - start + 1

        num = np.zeros((self.seq_len, self.n_features + 1), dtype=np.float32)
        cat = np.zeros(self.seq_len, dtype=np.int64)

        num[-n_real:, : self.n_features] = self.x_num[start : row + 1]
        num[-n_real:, self.n_features] = 1.0
        cat[-n_real:] = self.x_cat[start : row + 1]

        return torch.from_numpy(num), torch.from_numpy(cat), torch.tensor(
            self.y[row], dtype=torch.float32
        )
