"""
Feature engineering for sequential fraud detection.

The single most important property of this file is that every derived statistic
is CAUSAL: it is computed using only transactions that happened strictly before
the transaction being described. A rolling or expanding aggregate that includes
the current or a future transaction leaks information backwards in time and
silently inflates every downstream metric.

Run with:
    python -m src.features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# Columns kept from the raw CSVs.
#
# Dropped on purpose: first, last, gender, street, city, state, zip, job,
# trans_num, unix_time. These are identifiers or near-identifiers with no
# generalisable predictive signal. Keeping a per-cardholder identifier invites
# the model to memorise individuals rather than learn behaviour, and trans_num
# is a primary key that a tree model would happily overfit to.
USE_COLS = [
    "trans_date_trans_time",
    "cc_num",
    "category",
    "amt",
    "lat",
    "long",
    "city_pop",
    "dob",
    "merch_lat",
    "merch_long",
    "is_fraud",
]

NUMERIC_FEATURES = [
    "log_amt",
    "log_hours_since_prev",
    "hour_of_day",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "log_amt_vs_card_mean",
    "log_distance_km",
    "log_city_pop",
    "age",
]

# A gap of more than 30 days is treated as "no meaningful recent history".
MAX_GAP_HOURS = 720.0
# Ratio of current amount to the card's prior mean amount, clipped for stability.
MAX_AMT_RATIO = 100.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great circle distance in kilometres between two arrays of coordinates."""
    radius = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_raw() -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Load both CSVs and concatenate them.

    The two files are already separated by time, so the chronological split is
    defined by the last timestamp in fraudTrain.csv. Splitting by timestamp
    rather than by source file guarantees that every test row is strictly later
    than every training row, which is what makes the backward-looking sequence
    windows safe.
    """
    print("Loading CSVs ...")
    train = pd.read_csv(
        config.TRAIN_CSV,
        usecols=USE_COLS,
        parse_dates=["trans_date_trans_time", "dob"],
    )
    test = pd.read_csv(
        config.TEST_CSV,
        usecols=USE_COLS,
        parse_dates=["trans_date_trans_time", "dob"],
    )

    split_ts = train["trans_date_trans_time"].max()
    print(f"  fraudTrain.csv: {len(train):,} rows, "
          f"{train['trans_date_trans_time'].min()} to {train['trans_date_trans_time'].max()}")
    print(f"  fraudTest.csv:  {len(test):,} rows, "
          f"{test['trans_date_trans_time'].min()} to {test['trans_date_trans_time'].max()}")
    print(f"  Chronological split point: {split_ts}")

    df = pd.concat([train, test], ignore_index=True)
    return df, split_ts


def build_features(df: pd.DataFrame, split_ts: pd.Timestamp) -> pd.DataFrame:
    """Sort by card and time, then derive every feature causally."""
    print("Sorting by card and timestamp ...")
    # mergesort is stable, which keeps the ordering reproducible for ties.
    df = df.sort_values(
        ["cc_num", "trans_date_trans_time"], kind="mergesort"
    ).reset_index(drop=True)

    ts = df["trans_date_trans_time"]

    print("Deriving features ...")

    # Amount. Heavily right skewed, so log1p keeps the scale usable.
    df["log_amt"] = np.log1p(df["amt"])

    # Time since that card's previous transaction. This is the single most
    # important sequential feature: a burst of transactions on a card that
    # normally sees one a day is exactly the behavioural signal we want.
    # groupby.shift(1) looks strictly backwards, so this is causal by
    # construction. The first transaction of each card has no predecessor and
    # is assigned the maximum gap.
    prev_ts = df.groupby("cc_num", sort=False)["trans_date_trans_time"].shift(1)
    gap_hours = (ts - prev_ts).dt.total_seconds() / 3600.0
    gap_hours = gap_hours.fillna(MAX_GAP_HOURS).clip(lower=0.0, upper=MAX_GAP_HOURS)
    df["log_hours_since_prev"] = np.log1p(gap_hours)

    # Calendar features. Fraud in this dataset is concentrated in the small
    # hours, so hour of day carries real signal.
    df["hour_of_day"] = ts.dt.hour.astype("float64")
    df["day_of_week"] = ts.dt.dayofweek.astype("float64")
    # Cyclic encoding so that hour 23 sits next to hour 0 rather than 23 units
    # away. Trees can use the raw hour; the neural network benefits from this.
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour_of_day"] / 24.0)

    # Amount relative to what this card normally spends.
    #
    # Computed as a causal expanding mean that EXCLUDES the current row:
    #   prior_sum   = cumulative sum up to and including this row, minus this row
    #   prior_count = number of rows seen before this one for this card
    # The current transaction therefore never contributes to its own baseline.
    cum_amt = df.groupby("cc_num", sort=False)["amt"].cumsum()
    prior_sum = cum_amt - df["amt"]
    prior_count = df.groupby("cc_num", sort=False).cumcount().astype("float64")
    prior_mean = prior_sum / prior_count.where(prior_count > 0)
    ratio = df["amt"] / prior_mean
    # First transaction of a card has no baseline, so the ratio is defined as 1.
    ratio = ratio.fillna(1.0).clip(lower=0.0, upper=MAX_AMT_RATIO)
    df["log_amt_vs_card_mean"] = np.log1p(ratio)

    # Distance between the cardholder's home location and the merchant.
    dist = haversine_km(
        df["lat"].to_numpy(),
        df["long"].to_numpy(),
        df["merch_lat"].to_numpy(),
        df["merch_long"].to_numpy(),
    )
    df["log_distance_km"] = np.log1p(dist)

    df["log_city_pop"] = np.log1p(df["city_pop"])

    # Age at the time of the transaction, not age today.
    df["age"] = (ts - df["dob"]).dt.days / 365.25

    # Category is integer encoded and fed to an embedding layer.
    # Index 0 is reserved for the padding step, so real categories start at 1.
    cat_codes, cat_names = pd.factorize(df["category"], sort=True)
    df["category_idx"] = cat_codes.astype(np.int64) + 1

    df["is_test"] = (ts > split_ts).to_numpy()

    n_train = int((~df["is_test"]).sum())
    n_test = int(df["is_test"].sum())
    print(f"  Train rows: {n_train:,}  Test rows: {n_test:,}")
    print(f"  Categories: {len(cat_names)} -> embedding indices 1..{len(cat_names)}")

    df.attrs["category_names"] = list(cat_names)
    return df


def card_block_starts(cc_num: np.ndarray) -> np.ndarray:
    """
    For each row, the array index where that card's block of rows begins.

    The frame is sorted by card, so each card occupies one contiguous block.
    This array is what stops a sequence window from running off the start of a
    card and picking up the tail of a different card's history.
    """
    change = np.flatnonzero(np.r_[True, cc_num[1:] != cc_num[:-1]])
    block_lengths = np.diff(np.r_[change, len(cc_num)])
    return np.repeat(change, block_lengths)


def standardise(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """
    Z-score the numeric features using statistics from the TRAINING rows only.

    Fitting the scaler on the full dataset would be a real, if subtle, form of
    leakage: the test period's distribution would inform the training input.
    """
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0)
    std[std < 1e-8] = 1.0
    return ((x - mean) / std).astype(np.float32)


def main() -> None:
    config.ensure_dirs()

    df, split_ts = load_raw()
    df = build_features(df, split_ts)

    is_test = df["is_test"].to_numpy()
    train_mask = ~is_test

    x_num_raw = df[NUMERIC_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(x_num_raw).all():
        bad = np.argwhere(~np.isfinite(x_num_raw))
        raise ValueError(f"Non-finite feature values at {bad[:5]} (and possibly more)")

    x_num = standardise(x_num_raw, train_mask)
    x_cat = df["category_idx"].to_numpy(dtype=np.int64)
    y = df["is_fraud"].to_numpy(dtype=np.float32)
    amt = df["amt"].to_numpy(dtype=np.float32)
    block_start = card_block_starts(df["cc_num"].to_numpy())

    fraud_rate_train = float(y[train_mask].mean())
    fraud_rate_test = float(y[is_test].mean())
    print(f"  Fraud rate, train: {fraud_rate_train * 100:.3f} percent")
    print(f"  Fraud rate, test:  {fraud_rate_test * 100:.3f} percent")

    print(f"Writing {config.FEATURE_CACHE} ...")
    np.savez(
        config.FEATURE_CACHE,
        x_num=x_num,
        x_cat=x_cat,
        y=y,
        amt=amt,
        block_start=block_start,
        is_test=is_test,
        feature_names=np.array(NUMERIC_FEATURES),
        category_names=np.array([str(c) for c in df.attrs["category_names"]]),
        split_ts=np.array(str(split_ts)),
    )
    size_mb = config.FEATURE_CACHE.stat().st_size / 1e6
    print(f"Done. Cache is {size_mb:.1f} MB, "
          f"{x_num.shape[0]:,} rows by {x_num.shape[1]} numeric features.")


if __name__ == "__main__":
    main()
