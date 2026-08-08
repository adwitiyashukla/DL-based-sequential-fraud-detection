"""
Non sequential baselines: LightGBM and logistic regression.

A neural network with nothing to compare against is not a finding. These two
models see exactly the same engineered features as the GRU, but only for the
transaction being scored. They never see the preceding nine transactions.

Note the comparison is deliberately conservative. The feature vector already
contains hand crafted history summaries (hours_since_prev, amt_vs_card_mean),
so the baselines are not blind to the past. The question this isolates is
narrower and more precise: does modelling the sequence explicitly add anything
beyond those hand crafted summaries?

Run with:
    python -m src.baselines
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from src import config
from src.dataset import load_cache, split_indices


def one_hot(codes: np.ndarray, n_categories: int) -> np.ndarray:
    """Dense one hot encoding of the category index (1..n_categories)."""
    out = np.zeros((len(codes), n_categories), dtype=np.float32)
    out[np.arange(len(codes)), codes - 1] = 1.0
    return out


def train_lightgbm(x_tr, y_tr, x_te, cat_index: int) -> np.ndarray:
    import lightgbm as lgb

    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr) - n_pos)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        # Mirrors the pos_weight used in the GRU loss, so neither model gets an
        # advantage purely from how the class imbalance was handled.
        scale_pos_weight=n_neg / max(n_pos, 1.0),
        n_jobs=config.NUM_THREADS,
        random_state=config.SEED,
        verbose=-1,
    )
    model.fit(x_tr, y_tr, categorical_feature=[cat_index])
    # Persist the booster in LightGBM's own text format. It carries the
    # categorical feature handling with it, so the demo app can reload and
    # predict without reconstructing the sklearn wrapper.
    model.booster_.save_model(str(config.LGB_MODEL))
    return model.predict_proba(x_te)[:, 1]


def train_logreg(x_tr, y_tr, x_te) -> np.ndarray:
    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="lbfgs",
        random_state=config.SEED,
    )
    model.fit(x_tr, y_tr)
    return model.predict_proba(x_te)[:, 1]


def main() -> None:
    config.ensure_dirs()
    cache = load_cache()
    x_num, x_cat, y = cache["x_num"], cache["x_cat"], cache["y"]
    train_rows, test_rows = split_indices(cache["is_test"])
    n_categories = int(x_cat.max())

    y_tr = y[train_rows]
    y_te = y[test_rows]
    print(f"Train rows: {len(train_rows):,} ({int(y_tr.sum()):,} fraud)")
    print(f"Test rows:  {len(test_rows):,} ({int(y_te.sum()):,} fraud)")

    # LightGBM handles the category natively as a categorical split, so it goes
    # in as a single integer column rather than one hot.
    lgb_tr = np.column_stack([x_num[train_rows], x_cat[train_rows].astype(np.float32)])
    lgb_te = np.column_stack([x_num[test_rows], x_cat[test_rows].astype(np.float32)])
    cat_index = lgb_tr.shape[1] - 1

    print("\nTraining LightGBM ...")
    t0 = time.time()
    scores = train_lightgbm(lgb_tr, y_tr, lgb_te, cat_index)
    np.save(config.SCORES["lightgbm"], scores)
    print(f"  done in {time.time() - t0:.0f}s, wrote {config.SCORES['lightgbm'].name}")

    # Logistic regression is linear, so the category must be one hot encoded.
    lr_tr = np.column_stack([x_num[train_rows], one_hot(x_cat[train_rows], n_categories)])
    lr_te = np.column_stack([x_num[test_rows], one_hot(x_cat[test_rows], n_categories)])

    print("\nTraining logistic regression ...")
    t0 = time.time()
    scores = train_logreg(lr_tr, y_tr, lr_te)
    np.save(config.SCORES["logreg"], scores)
    print(f"  done in {time.time() - t0:.0f}s, wrote {config.SCORES['logreg'].name}")


if __name__ == "__main__":
    main()
