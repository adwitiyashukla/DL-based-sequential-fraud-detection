"""
Evaluation, the way a fraud team would actually do it.

Three deliberate choices here, each of which is a departure from the typical
student fraud project:

1. PR-AUC leads, ROC-AUC is reported but demoted. At a 0.4 percent base rate
   the negative class dominates the false positive rate, so ROC-AUC looks
   impressive for models that are not useful.
2. Recall at a fixed alert budget. A review team has finite capacity. The
   question that decides a purchase is "if we can review the top 1 percent of
   transactions, how much fraud do we catch", not "what is the F1 score".
3. The decision threshold is chosen by minimising expected cost, not by
   defaulting to 0.5. A missed fraud costs the transaction amount, a false
   positive costs a fixed review expense.

Run with:
    python -m src.evaluate
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # No display on a headless run; write straight to PNG.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src import config
from src.dataset import load_cache, split_indices


def cost_analysis(y: np.ndarray, scores: np.ndarray, amt: np.ndarray) -> dict:
    """
    Sweep every possible alert cutoff and find the one that minimises total cost.

    Working in "top k transactions" space rather than looping over candidate
    thresholds makes this exact and vectorised: sorting once by descending score
    means the cumulative sums give the confusion matrix at every cutoff at once.
    """
    order = np.argsort(-scores, kind="stable")
    y_s = y[order].astype(np.float64)
    amt_s = amt[order].astype(np.float64)
    s_s = scores[order].astype(np.float64)

    n = len(y_s)
    total_fraud_amt = float((y_s * amt_s).sum())

    k = np.arange(n + 1, dtype=np.float64)
    tp_k = np.concatenate([[0.0], np.cumsum(y_s)])
    caught_amt_k = np.concatenate([[0.0], np.cumsum(y_s * amt_s)])
    fp_k = k - tp_k

    # Cost of reviewing k transactions: the review expense for every false
    # positive, plus the full value of every fraud that was not flagged.
    cost = config.REVIEW_COST * fp_k + (total_fraud_amt - caught_amt_k)

    best_k = int(np.argmin(cost))
    best_threshold = 1.0 + 1e-9 if best_k == 0 else float(s_s[best_k - 1])

    # Recompute the confusion matrix from the threshold itself so that the
    # reported numbers are internally consistent even when scores are tied.
    pred = scores >= best_threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    best_cost = config.REVIEW_COST * fp + float(amt[(y == 1) & (~pred)].sum())

    naive = scores >= 0.5
    naive_fp = int(((naive == 1) & (y == 0)).sum())
    naive_cost = config.REVIEW_COST * naive_fp + float(amt[(y == 1) & (~naive)].sum())

    return {
        "threshold": best_threshold,
        "alert_rate": best_k / n,
        "cost": best_cost,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "naive_threshold": 0.5,
        "naive_cost": naive_cost,
        "naive_alerts": int(naive.sum()),
        "saving_vs_naive": naive_cost - best_cost,
        "total_fraud_amount": total_fraud_amt,
        "curve_thresholds": s_s,
        "curve_costs": cost[1:],
    }


def budget_metrics(y: np.ndarray, scores: np.ndarray, budget: float) -> dict:
    """Recall and precision if the team can only review the top `budget` fraction."""
    n = len(scores)
    k = max(1, int(round(budget * n)))
    order = np.argsort(-scores, kind="stable")[:k]
    tp = float(y[order].sum())
    total_pos = float(y.sum())
    return {
        "alerts_reviewed": k,
        "recall": tp / total_pos,
        "precision": tp / k,
    }


def evaluate_model(y, scores, amt) -> dict:
    out = {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }
    for b in config.ALERT_BUDGETS:
        out[f"recall_at_{b:g}"] = budget_metrics(y, scores, b)
    out["cost"] = cost_analysis(y, scores, amt)
    return out


def plot_pr_curves(y, all_scores, results) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, scores in all_scores.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision, linewidth=1.8,
                label=f"{config.MODEL_LABELS[name]}  (AP = {results[name]['pr_auc']:.3f})")
    base = float(y.mean())
    ax.axhline(base, linestyle="--", color="grey", linewidth=1.0,
               label=f"Random classifier ({base:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on the held out time period")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)


def plot_cost_curves(results) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, res in results.items():
        c = res["cost"]
        thr = c["curve_thresholds"]
        costs = c["curve_costs"]
        step = max(1, len(thr) // 4000)  # Downsample purely for plot size.
        ax.plot(thr[::step], costs[::step], linewidth=1.6,
                label=config.MODEL_LABELS[name])
        ax.plot([c["threshold"]], [c["cost"]], marker="o", markersize=8,
                color=ax.lines[-1].get_color(), zorder=5)
        ax.annotate(
            f"{config.MODEL_LABELS[name].split(' (')[0]}\n"
            f"t = {c['threshold']:.3f}, ${c['cost'] / 1000:,.0f}k",
            xy=(c["threshold"], c["cost"]),
            xytext=(9, 9), textcoords="offset points", fontsize=8,
        )
    ax.axvline(0.5, linestyle="--", color="grey", linewidth=1.0,
               label="Naive threshold (0.5)")
    # Log scale, otherwise logistic regression's cost range flattens the two
    # models the comparison is actually about.
    ax.set_yscale("log")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Total cost in dollars (log scale)")
    ax.set_xlim(-0.03, 1.12)
    ax.set_title(
        f"Expected cost vs threshold\n"
        f"(missed fraud costs the transaction amount, "
        f"a false positive costs ${config.REVIEW_COST:.0f})",
        fontsize=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "cost_curve.png", dpi=150)
    plt.close(fig)


def plot_confusion(results) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        c = res["cost"]["confusion"]
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
        ax.imshow(np.log1p(m), cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(m[i, j]):,}", ha="center", va="center",
                        fontsize=11,
                        color="white" if np.log1p(m[i, j]) > np.log1p(m).max() * 0.6 else "black")
        ax.set_xticks([0, 1], ["Predicted\nlegitimate", "Predicted\nfraud"], fontsize=8)
        ax.set_yticks([0, 1], ["Actual\nlegitimate", "Actual\nfraud"], fontsize=8)
        ax.set_title(f"{config.MODEL_LABELS[name]}\nthreshold = {res['cost']['threshold']:.3f}",
                     fontsize=9)
    fig.suptitle("Confusion matrices at the cost minimising threshold", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_loss_curve() -> None:
    step_path = config.CACHE_DIR / "gru_step_losses.npy"
    if not step_path.exists():
        return
    steps = np.load(step_path)
    epochs = np.load(config.CACHE_DIR / "gru_epoch_losses.npy")
    per_epoch = max(1, len(steps) // max(1, len(epochs)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, len(steps) + 1) / per_epoch, steps,
            linewidth=1.4, label="Training loss (running mean)")
    ax.plot(np.arange(1, len(epochs) + 1), epochs, marker="o",
            linewidth=1.6, label="Epoch mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted BCE loss")
    ax.set_title("GRU training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "loss_curve.png", dpi=150)
    plt.close(fig)


def print_table(results, y) -> None:
    n = len(y)
    pos = int(y.sum())
    trivial_accuracy = (n - pos) / n

    print("\n" + "=" * 96)
    print(f"Test set: {n:,} transactions, {pos:,} fraudulent "
          f"({pos / n * 100:.3f} percent)")
    print(f"A model that predicts 'legitimate' for every transaction scores "
          f"{trivial_accuracy * 100:.3f} percent accuracy.")
    print("This is why accuracy is not reported as a headline metric.")
    print("=" * 96)

    header = (f"{'Model':<26}{'PR-AUC':>9}{'ROC-AUC':>10}"
              f"{'R@0.1%':>9}{'R@1%':>9}{'Cost':>13}{'vs naive 0.5':>15}")
    print(header)
    print("-" * 96)
    for name, res in results.items():
        c = res["cost"]
        print(
            f"{config.MODEL_LABELS[name]:<26}"
            f"{res['pr_auc']:>9.4f}"
            f"{res['roc_auc']:>10.4f}"
            f"{res['recall_at_0.001']['recall']:>9.3f}"
            f"{res['recall_at_0.01']['recall']:>9.3f}"
            f"{'$' + format(c['cost'], ',.0f'):>13}"
            f"{'$' + format(c['saving_vs_naive'], ',.0f') + ' saved':>15}"
        )
    print("-" * 96)
    best = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"Highest PR-AUC: {config.MODEL_LABELS[best]}")
    print("=" * 96 + "\n")


def to_serialisable(results: dict) -> dict:
    """Strip the large plotting arrays before writing metrics.json."""
    out = {}
    for name, res in results.items():
        c = {k: v for k, v in res["cost"].items()
             if k not in ("curve_thresholds", "curve_costs")}
        out[name] = {k: v for k, v in res.items() if k != "cost"}
        out[name]["cost"] = c
    return out


def main() -> None:
    config.ensure_dirs()
    cache = load_cache()
    _, test_rows = split_indices(cache["is_test"])
    y = cache["y"][test_rows]
    amt = cache["amt"][test_rows]

    all_scores = {}
    for name, path in config.SCORES.items():
        if path.exists():
            s = np.load(path)
            if len(s) != len(y):
                raise ValueError(
                    f"{path.name} has {len(s):,} scores but the test set has {len(y):,} rows."
                )
            all_scores[name] = s
        else:
            print(f"Skipping {name}: {path.name} not found.")

    if not all_scores:
        raise SystemExit("No score files found. Run src.train and src.baselines first.")

    results = {name: evaluate_model(y, s, amt) for name, s in all_scores.items()}

    print_table(results, y)

    plot_pr_curves(y, all_scores, results)
    plot_cost_curves(results)
    plot_confusion(results)
    plot_loss_curve()

    payload = {
        "test_transactions": int(len(y)),
        "test_frauds": int(y.sum()),
        "test_fraud_rate": float(y.mean()),
        "trivial_all_negative_accuracy": float((len(y) - y.sum()) / len(y)),
        "review_cost_per_false_positive": config.REVIEW_COST,
        "sequence_length": config.SEQ_LEN,
        "models": to_serialisable(results),
    }
    with open(config.RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote plots and metrics.json to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
