from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from src import config
from src.dataset import load_cache, split_indices
from src.features import load_raw, build_features

SPACE_DIR = config.ROOT / "space"
ASSETS = SPACE_DIR / "assets"

PER_SCENARIO = 50

SCENARIO_ORDER = [
    "Fraud caught by the GRU, missed by LightGBM",
    "Fraud caught by both models",
    "Fraud missed by the GRU",
    "False alarm raised by the GRU",
    "Legitimate, correctly cleared",
]


def load_thresholds() -> tuple[float, float]:
    with open(config.RESULTS_DIR / "metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    return (
        float(m["models"]["gru"]["cost"]["threshold"]),
        float(m["models"]["lightgbm"]["cost"]["threshold"]),
    )


def load_headline_metrics() -> dict:
    with open(config.RESULTS_DIR / "metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    out = {}
    for name in ("gru", "lightgbm", "logreg"):
        out[name] = {
            "pr_auc": m["models"][name]["pr_auc"],
            "roc_auc": m["models"][name]["roc_auc"],
            "recall_at_0.001": m["models"][name]["recall_at_0.001"]["recall"],
            "recall_at_0.01": m["models"][name]["recall_at_0.01"]["recall"],
        }
    return out


def build_scenarios(y_te, s_gru, s_lgb, thr_gru, thr_lgb) -> dict[str, np.ndarray]:
    fraud = y_te == 1
    legit = y_te == 0
    gru_flags = s_gru >= thr_gru
    lgb_flags = s_lgb >= thr_lgb

    return {
        SCENARIO_ORDER[0]: fraud & gru_flags & ~lgb_flags,
        SCENARIO_ORDER[1]: fraud & gru_flags & lgb_flags,
        SCENARIO_ORDER[2]: fraud & ~gru_flags,
        SCENARIO_ORDER[3]: legit & gru_flags,
        SCENARIO_ORDER[4]: legit & (s_gru < 0.02) & (s_lgb < 0.02),
    }


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)

    cache = load_cache()
    x_num, x_cat, y = cache["x_num"], cache["x_cat"], cache["y"]
    amt, block_start, is_test = cache["amt"], cache["block_start"], cache["is_test"]
    _, test_rows = split_indices(is_test)

    s_gru = np.load(config.SCORES["gru"])
    s_lgb = np.load(config.SCORES["lightgbm"])
    y_te = y[test_rows]
    amt_te = amt[test_rows]

    thr_gru, thr_lgb = load_thresholds()
    print(f"Thresholds: GRU {thr_gru:.4f}, LightGBM {thr_lgb:.4f}")

    scenarios = build_scenarios(y_te, s_gru, s_lgb, thr_gru, thr_lgb)
    picked: list[int] = []
    labels: list[str] = []
    for name in SCENARIO_ORDER:
        mask = scenarios[name]
        idx = np.flatnonzero(mask)
        take = idx if len(idx) <= PER_SCENARIO else rng.choice(idx, PER_SCENARIO, replace=False)
        take = np.sort(take)
        picked.extend(take.tolist())
        labels.extend([name] * len(take))
        print(f"  {name}: {len(idx):,} available, {len(take)} exported")

    picked = np.array(picked, dtype=np.int64)
    global_rows = test_rows[picked]
    n_samples = len(picked)
    seq = config.SEQ_LEN
    n_feat = x_num.shape[1]

    print("Re-deriving human readable values from the raw CSVs ...")
    df, split_ts = load_raw()
    df = build_features(df, split_ts)
    if len(df) != len(y):
        raise RuntimeError("Row count mismatch: the cache and the CSVs disagree.")

    disp_ts = df["trans_date_trans_time"].to_numpy()
    disp_amt = df["amt"].to_numpy(dtype=np.float64)
    disp_cat = df["category"].astype(str).to_numpy()
    disp_card = df["cc_num"].to_numpy()
    disp_dist = np.expm1(df["log_distance_km"].to_numpy(dtype=np.float64))
    disp_gap = np.expm1(df["log_hours_since_prev"].to_numpy(dtype=np.float64))
    disp_ratio = np.expm1(df["log_amt_vs_card_mean"].to_numpy(dtype=np.float64))
    del df

    win_num = np.zeros((n_samples, seq, n_feat + 1), dtype=np.float32)
    win_cat = np.zeros((n_samples, seq), dtype=np.int64)
    records = []

    for k, r in enumerate(global_rows):
        r = int(r)
        start = max(int(block_start[r]), r - seq + 1)
        n_real = r - start + 1

        win_num[k, -n_real:, :n_feat] = x_num[start : r + 1]
        win_num[k, -n_real:, n_feat] = 1.0
        win_cat[k, -n_real:] = x_cat[start : r + 1]

        for pos, row in enumerate(range(start, r + 1), start=1):
            records.append(
                {
                    "sample_id": k,
                    "position": pos,
                    "is_target": row == r,
                    "timestamp": pd.Timestamp(disp_ts[row]).strftime("%Y-%m-%d %H:%M"),
                    "amount": round(float(disp_amt[row]), 2),
                    "category": disp_cat[row].replace("_", " "),
                    "distance_km": round(float(disp_dist[row]), 1),
                    "hours_since_prev": round(float(disp_gap[row]), 1),
                    "amt_vs_card_mean": round(float(disp_ratio[row]), 2),
                }
            )

    display = pd.DataFrame.from_records(records)

    meta = pd.DataFrame(
        {
            "sample_id": np.arange(n_samples),
            "scenario": labels,
            "card": [f"card ending {str(int(c))[-4:]}" for c in disp_card[global_rows]],
            "is_fraud": y_te[picked].astype(int),
            "score_gru": s_gru[picked].astype(np.float32),
            "score_lgb": s_lgb[picked].astype(np.float32),
            "amount": np.round(disp_amt[global_rows], 2),
            "history_length": [
                int(min(seq, int(r) - max(int(block_start[int(r)]), int(r) - seq + 1) + 1))
                for r in global_rows
            ],
        }
    )

    np.savez_compressed(
        ASSETS / "demo_windows.npz", win_num=win_num, win_cat=win_cat
    )
    display.to_csv(ASSETS / "demo_display.csv", index=False)
    meta.to_csv(ASSETS / "demo_meta.csv", index=False)

    np.savez_compressed(
        ASSETS / "test_scores.npz",
        gru=s_gru.astype(np.float32),
        lgb=s_lgb.astype(np.float32),
        y=y_te.astype(np.int8),
        amt=amt_te.astype(np.float32),
    )

    shutil.copy2(config.GRU_CHECKPOINT, ASSETS / "gru.pth")
    shutil.copy2(config.LGB_MODEL, ASSETS / "lightgbm.txt")

    config_payload = {
        "seq_len": seq,
        "n_numeric": n_feat + 1,
        "n_categories": int(x_cat.max()),
        "threshold_gru": thr_gru,
        "threshold_lgb": thr_lgb,
        "default_review_cost": config.REVIEW_COST,
        "n_test": int(len(y_te)),
        "n_fraud": int(y_te.sum()),
        "total_fraud_amount": float((amt_te * y_te).sum()),
        "trivial_accuracy": float((len(y_te) - y_te.sum()) / len(y_te)),
        "train_rows": int((~is_test).sum()),
        "train_frauds": int(y[~is_test].sum()),
        "scenario_order": SCENARIO_ORDER,
        "metrics": load_headline_metrics(),
    }
    with open(ASSETS / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    total_mb = sum(p.stat().st_size for p in ASSETS.iterdir()) / 1e6
    print(f"\nExported {n_samples} demo cases to {ASSETS}")
    print(f"Payload size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
