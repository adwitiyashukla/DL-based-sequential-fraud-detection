from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

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

MAX_GAP_HOURS = 720.0
MAX_AMT_RATIO = 100.0


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_raw() -> tuple[pd.DataFrame, pd.Timestamp]:
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
    print("Sorting by card and timestamp ...")
    df = df.sort_values(
        ["cc_num", "trans_date_trans_time"], kind="mergesort"
    ).reset_index(drop=True)

    ts = df["trans_date_trans_time"]

    print("Deriving features ...")

    df["log_amt"] = np.log1p(df["amt"])

    prev_ts = df.groupby("cc_num", sort=False)["trans_date_trans_time"].shift(1)
    gap_hours = (ts - prev_ts).dt.total_seconds() / 3600.0
    gap_hours = gap_hours.fillna(MAX_GAP_HOURS).clip(lower=0.0, upper=MAX_GAP_HOURS)
    df["log_hours_since_prev"] = np.log1p(gap_hours)

    df["hour_of_day"] = ts.dt.hour.astype("float64")
    df["day_of_week"] = ts.dt.dayofweek.astype("float64")
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour_of_day"] / 24.0)

    cum_amt = df.groupby("cc_num", sort=False)["amt"].cumsum()
    prior_sum = cum_amt - df["amt"]
    prior_count = df.groupby("cc_num", sort=False).cumcount().astype("float64")
    prior_mean = prior_sum / prior_count.where(prior_count > 0)
    ratio = df["amt"] / prior_mean
    ratio = ratio.fillna(1.0).clip(lower=0.0, upper=MAX_AMT_RATIO)
    df["log_amt_vs_card_mean"] = np.log1p(ratio)

    dist = haversine_km(
        df["lat"].to_numpy(),
        df["long"].to_numpy(),
        df["merch_lat"].to_numpy(),
        df["merch_long"].to_numpy(),
    )
    df["log_distance_km"] = np.log1p(dist)

    df["log_city_pop"] = np.log1p(df["city_pop"])

    df["age"] = (ts - df["dob"]).dt.days / 365.25

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
    change = np.flatnonzero(np.r_[True, cc_num[1:] != cc_num[:-1]])
    block_lengths = np.diff(np.r_[change, len(cc_num)])
    return np.repeat(change, block_lengths)


def standardise(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
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
