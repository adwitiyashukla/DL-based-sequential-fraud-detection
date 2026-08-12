from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from model import GRUFraudModel

ASSETS = Path(__file__).parent / "assets"

with open(ASSETS / "config.json", encoding="utf-8") as f:
    CFG = json.load(f)

SEQ_LEN = CFG["seq_len"]
N_NUMERIC = CFG["n_numeric"]
N_FEAT = N_NUMERIC - 1
THR_GRU = CFG["threshold_gru"]
THR_LGB = CFG["threshold_lgb"]

_w = np.load(ASSETS / "demo_windows.npz")
WIN_NUM = _w["win_num"]
WIN_CAT = _w["win_cat"]

META = pd.read_csv(ASSETS / "demo_meta.csv")
DISPLAY = pd.read_csv(ASSETS / "demo_display.csv")

_s = np.load(ASSETS / "test_scores.npz")
S_GRU = _s["gru"].astype(np.float64)
S_LGB = _s["lgb"].astype(np.float64)
Y_TEST = _s["y"].astype(np.float64)
AMT_TEST = _s["amt"].astype(np.float64)

N_TEST = len(Y_TEST)
N_FRAUD = int(Y_TEST.sum())
TOTAL_FRAUD_AMT = float((Y_TEST * AMT_TEST).sum())

GRU = GRUFraudModel(n_numeric=N_NUMERIC, n_categories=CFG["n_categories"])
GRU.load_state_dict(torch.load(ASSETS / "gru.pth", map_location="cpu", weights_only=True))
GRU.eval()
torch.set_num_threads(2)

BOOSTER = lgb.Booster(model_file=str(ASSETS / "lightgbm.txt"))

RNG = np.random.default_rng()


def score_gru(sample_id: int) -> float:
    with torch.no_grad():
        logit = GRU(
            torch.from_numpy(WIN_NUM[sample_id : sample_id + 1]),
            torch.from_numpy(WIN_CAT[sample_id : sample_id + 1]),
        )
        return float(torch.sigmoid(logit).item())


def score_lgb(sample_id: int) -> float:
    flat = np.concatenate(
        [
            WIN_NUM[sample_id, -1, :N_FEAT].astype(np.float64),
            [float(WIN_CAT[sample_id, -1])],
        ]
    ).reshape(1, -1)
    return float(BOOSTER.predict(flat)[0])


def _cumulative(scores: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    ys = Y_TEST[order]
    return {
        "s_desc": scores[order],
        "tp": np.concatenate([[0.0], np.cumsum(ys)]),
        "famt": np.concatenate([[0.0], np.cumsum(ys * AMT_TEST[order])]),
    }


CUM = {"GRU": _cumulative(S_GRU), "LightGBM": _cumulative(S_LGB)}
K_AXIS = np.arange(N_TEST + 1, dtype=np.float64)


def cost_curve(model: str, review_cost: float) -> np.ndarray:
    c = CUM[model]
    return review_cost * (K_AXIS - c["tp"]) + (TOTAL_FRAUD_AMT - c["famt"])


def optimal_threshold(model: str, review_cost: float) -> tuple[float, float]:
    curve = cost_curve(model, review_cost)
    k = int(np.argmin(curve))
    thr = 1.0 if k == 0 else float(CUM[model]["s_desc"][k - 1])
    return thr, float(curve[k])


def confusion_at(model: str, threshold: float, review_cost: float) -> dict:
    c = CUM[model]
    k = int(np.searchsorted(-c["s_desc"], -threshold, side="right"))
    tp = float(c["tp"][k])
    fp = k - tp
    fn = N_FRAUD - tp
    tn = N_TEST - k - fn
    cost = review_cost * fp + (TOTAL_FRAUD_AMT - float(c["famt"][k]))
    return {
        "alerts": k,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "cost": cost,
        "recall": tp / N_FRAUD if N_FRAUD else 0.0,
        "precision": tp / k if k else 0.0,
        "alert_rate": k / N_TEST,
    }


def _history_table(sample_id: int) -> str:
    rows = DISPLAY[DISPLAY.sample_id == sample_id].sort_values("position")
    body = []
    for _, r in rows.iterrows():
        cls = "target" if bool(r.is_target) else ""
        gap = "first seen" if r.hours_since_prev >= 719 else f"{r.hours_since_prev:,.1f} h"
        body.append(
            f"<tr class='{cls}'>"
            f"<td>{r.timestamp}</td>"
            f"<td class='num'>${r.amount:,.2f}</td>"
            f"<td>{r.category}</td>"
            f"<td class='num'>{r.distance_km:,.0f} km</td>"
            f"<td class='num'>{gap}</td>"
            f"<td class='num'>{r.amt_vs_card_mean:,.2f}x</td>"
            f"</tr>"
        )
    return (
        "<div class='tablewrap'><table class='hist'>"
        "<thead><tr><th>Timestamp</th><th>Amount</th><th>Category</th>"
        "<th>Distance</th><th>Since previous</th><th>vs card average</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
        "<p class='caption'>The highlighted row is the transaction being scored. "
        "The rows above it are the context the GRU reads.</p>"
    )


def _score_panel(name: str, prob: float, threshold: float, subtitle: str) -> str:
    flagged = prob >= threshold
    pct = prob * 100
    tone = "flag" if flagged else "clear"
    verdict = "FLAG FOR REVIEW" if flagged else "ALLOW"
    return (
        f"<div class='scorecard {tone}'>"
        f"<div class='sc-name'>{name}</div>"
        f"<div class='sc-sub'>{subtitle}</div>"
        f"<div class='sc-prob'>{pct:.1f}<span>%</span></div>"
        f"<div class='sc-bar'><div class='sc-fill' style='width:{min(pct, 100):.1f}%'></div>"
        f"<div class='sc-thr' style='left:{threshold * 100:.1f}%'></div></div>"
        f"<div class='sc-verdict'>{verdict}</div>"
        f"<div class='sc-thrlab'>threshold {threshold:.3f}</div>"
        f"</div>"
    )


def _outcome_banner(is_fraud: int, p_gru: float, p_lgb: float) -> str:
    gru_flag = p_gru >= THR_GRU
    lgb_flag = p_lgb >= THR_LGB
    truth = "FRAUDULENT" if is_fraud else "LEGITIMATE"

    if is_fraud and gru_flag and not lgb_flag:
        tone, msg = "good", "The sequence model caught it. The flat model did not."
    elif is_fraud and gru_flag:
        tone, msg = "good", "Both models caught it."
    elif is_fraud and not gru_flag:
        tone, msg = "bad", "Both models missed this one. It is one of the 31 the GRU lets through."
    elif not is_fraud and gru_flag:
        tone, msg = "warn", "A false alarm. This costs a review, not a chargeback."
    else:
        tone, msg = "good", "Correctly cleared, with no analyst time spent."

    return (
        f"<div class='banner {tone}'>"
        f"<span class='b-label'>Ground truth</span>"
        f"<span class='b-truth'>{truth}</span>"
        f"<span class='b-msg'>{msg}</span></div>"
    )


def load_case(scenario: str):
    pool = META[META.scenario == scenario]
    if pool.empty:
        pool = META
    row = pool.iloc[int(RNG.integers(len(pool)))]
    sid = int(row.sample_id)

    p_gru = score_gru(sid)
    p_lgb = score_lgb(sid)

    header = (
        f"<div class='caseheader'>"
        f"<div><span class='ch-label'>Card</span><span class='ch-val'>{row.card}</span></div>"
        f"<div><span class='ch-label'>Transaction</span>"
        f"<span class='ch-val'>${row.amount:,.2f}</span></div>"
        f"<div><span class='ch-label'>History available</span>"
        f"<span class='ch-val'>{int(row.history_length)} of {SEQ_LEN} steps</span></div>"
        f"</div>"
    )
    panels = (
        "<div class='panelrow'>"
        + _score_panel("GRU", p_gru, THR_GRU, "reads the last 10 transactions")
        + _score_panel("LightGBM", p_lgb, THR_LGB, "reads this transaction only")
        + "</div>"
    )
    banner = _outcome_banner(int(row.is_fraud), p_gru, p_lgb)
    return header, _history_table(sid), panels, banner


def _metric_tile(label: str, value: str, sub: str = "") -> str:
    return (
        f"<div class='tile'><div class='t-label'>{label}</div>"
        f"<div class='t-value'>{value}</div><div class='t-sub'>{sub}</div></div>"
    )


def _style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        margin=dict(l=68, r=20, t=34, b=50),
    )
    grid = "rgba(148,163,184,0.20)"
    line = "rgba(148,163,184,0.35)"
    fig.update_xaxes(gridcolor=grid, zerolinecolor=line, linecolor=line)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=line, linecolor=line)
    return fig


def explore(review_cost: float, threshold: float):
    gru = confusion_at("GRU", threshold, review_cost)
    opt_thr, opt_cost = optimal_threshold("GRU", review_cost)
    opt_thr_l, opt_cost_l = optimal_threshold("LightGBM", review_cost)
    naive = confusion_at("GRU", 0.5, review_cost)

    tiles = (
        "<div class='tilehead'>GRU, at your chosen threshold</div>"
        "<div class='tilerow'>"
        + _metric_tile("Alerts raised", f"{gru['alerts']:,}",
                       f"{gru['alert_rate'] * 100:.2f}% of transactions")
        + _metric_tile("Fraud caught", f"{gru['tp']:,}",
                       f"{gru['recall'] * 100:.1f}% recall")
        + _metric_tile("Fraud missed", f"{gru['fn']:,}", "escaped review")
        + _metric_tile("False alarms", f"{gru['fp']:,}",
                       f"{gru['precision'] * 100:.1f}% precision")
        + _metric_tile("Total cost", f"${gru['cost']:,.0f}",
                       f"vs ${naive['cost']:,.0f} at threshold 0.5")
        + "</div>"
        + f"<div class='optnote'>At ${review_cost:,.2f} per review, each model's own cost "
          f"minimising threshold is <b>{opt_thr:.3f}</b> for the GRU "
          f"(<b>${opt_cost:,.0f}</b>) and <b>{opt_thr_l:.3f}</b> for LightGBM "
          f"(<b>${opt_cost_l:,.0f}</b>). Sequence context is worth "
          f"<b>${opt_cost_l - opt_cost:,.0f}</b>.</div>"
    )

    fig = go.Figure()
    for name, colour in (("GRU", "#4f46e5"), ("LightGBM", "#f59e0b")):
        curve = cost_curve(name, review_cost)
        s = CUM[name]["s_desc"]
        step = max(1, len(s) // 1500)
        fig.add_trace(
            go.Scatter(
                x=s[::step], y=curve[1:][::step], mode="lines", name=name,
                line=dict(color=colour, width=2),
                hovertemplate="threshold %{x:.3f}<br>cost $%{y:,.0f}<extra></extra>",
            )
        )
    fig.add_vline(x=threshold, line_dash="dash", line_color="#94a3b8",
                  annotation_text="your threshold", annotation_position="top")
    fig.add_trace(
        go.Scatter(x=[opt_thr], y=[opt_cost], mode="markers", name="GRU optimum",
                   marker=dict(color="#4f46e5", size=12, symbol="circle"),
                   hovertemplate="optimum %{x:.3f}<br>$%{y:,.0f}<extra></extra>")
    )
    fig.update_layout(
        yaxis_type="log",
        xaxis_title="Decision threshold",
        yaxis_title="Total cost, dollars (log scale)",
        height=420,
        hovermode="x unified",
    )
    fig.update_yaxes(dtick=1, tickprefix="$", tickformat="~s")
    return tiles, _style_fig(fig)


def snap_to_optimal(review_cost: float):
    thr, _ = optimal_threshold("GRU", review_cost)
    return thr


def pr_figure() -> go.Figure:
    fig = go.Figure()
    for name, colour in (("GRU", "#4f46e5"), ("LightGBM", "#f59e0b")):
        c = CUM[name]
        k = np.arange(1, N_TEST + 1)
        recall = c["tp"][1:] / N_FRAUD
        precision = c["tp"][1:] / k
        step = max(1, N_TEST // 2000)
        fig.add_trace(
            go.Scatter(x=recall[::step], y=precision[::step], mode="lines", name=name,
                       line=dict(color=colour, width=2)))
    fig.add_hline(y=N_FRAUD / N_TEST, line_dash="dot", line_color="#94a3b8",
                  annotation_text="random classifier")
    fig.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        yaxis_range=[0, 1.02], height=400,
    )
    return _style_fig(fig)


M = CFG["metrics"]
ABOUT = f"""
### What this is

Most credit card fraud models score each transaction in isolation. But fraud is a
behavioural signal: what matters is that *this card* has never behaved this way before.
This model builds a per card sequence of recent transactions and feeds it to a GRU.

The comparison is against a LightGBM model given the **identical features** for the
transaction being scored, but no sequence context. That isolates what the sequence adds.

### Results on the held out time period

{CFG['n_test']:,} transactions, {CFG['n_fraud']:,} fraudulent ({CFG['n_fraud'] / CFG['n_test'] * 100:.3f} percent).

| Model | PR-AUC | ROC-AUC | Recall @ 0.1% | Recall @ 1% |
|---|---|---|---|---|
| **GRU (sequence)** | **{M['gru']['pr_auc']:.4f}** | {M['gru']['roc_auc']:.4f} | {M['gru']['recall_at_0.001']:.3f} | **{M['gru']['recall_at_0.01']:.3f}** |
| LightGBM (no sequence) | {M['lightgbm']['pr_auc']:.4f} | {M['lightgbm']['roc_auc']:.4f} | {M['lightgbm']['recall_at_0.001']:.3f} | {M['lightgbm']['recall_at_0.01']:.3f} |
| Logistic regression | {M['logreg']['pr_auc']:.4f} | {M['logreg']['roc_auc']:.4f} | {M['logreg']['recall_at_0.001']:.3f} | {M['logreg']['recall_at_0.01']:.3f} |

Accuracy is deliberately absent. Predicting "legitimate" for every transaction scores
**{CFG['trivial_accuracy'] * 100:.3f} percent** while catching nothing.

### How it was built

- **Chronological split by timestamp.** Every test row is strictly later than every training row.
- **Causal features only.** A card's average spend is an expanding mean over prior rows, so a
  transaction never contributes to its own baseline.
- **Windows sliced on demand.** One flat array per split rather than materialising 1.29M
  sequences, which keeps the whole thing inside 16 GB.
- **Trained on CPU.** 22,577 parameters, 4 epochs, 7.1 minutes on a laptop with no GPU.

### Honest limits

The data is simulated (Sparkov), and rule generated fraud is far more learnable than the
adversarial kind, so a PR-AUC of 0.965 is not a production number. The cost minimising
threshold is also selected on the test set, which makes those dollar figures optimistic.
Both points are covered in more detail in the repository.

[Full code, methodology and limitations on GitHub](https://github.com/adwitiyashukla/DL-based-sequential-fraud-detection)
"""

CSS = """
.gradio-container { max-width: 1180px !important; }
#hero { padding: 4px 0 2px 0; }
#hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.02em;
  color: var(--body-text-color); }
#hero p { color: var(--body-text-color-subdued); margin: 0; font-size: 1.02rem; }

.caseheader { display:flex; gap:34px; padding:14px 18px;
  background: var(--background-fill-secondary);
  border:1px solid var(--border-color-primary); border-radius:10px;
  margin-bottom:6px; flex-wrap:wrap; }
.caseheader > div { display:flex; flex-direction:column; }
.ch-label { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.07em;
  color: var(--body-text-color-subdued); }
.ch-val { font-size:1.12rem; font-weight:650; color: var(--body-text-color); }

.tablewrap { overflow-x:auto; border:1px solid var(--border-color-primary); border-radius:10px; }
table.hist { width:100%; border-collapse:collapse; font-size:0.87rem; }
table.hist th { background: var(--background-fill-secondary); text-align:left;
  padding:9px 11px; font-weight:600; color: var(--body-text-color-subdued);
  border-bottom:1px solid var(--border-color-primary); white-space:nowrap;
  font-size:0.78rem; text-transform:uppercase; letter-spacing:0.04em; }
table.hist td { padding:8px 11px; border-bottom:1px solid var(--border-color-primary);
  color: var(--body-text-color); white-space:nowrap; }
table.hist td.num { text-align:right; font-variant-numeric:tabular-nums; }
table.hist tr.target td { background: rgba(99,102,241,0.18); font-weight:700; }
.caption { color: var(--body-text-color-subdued); font-size:0.8rem; margin:7px 2px 0 2px; }

.panelrow { display:flex; gap:16px; flex-wrap:wrap; }
.scorecard { flex:1; min-width:250px; border:1px solid var(--border-color-primary);
  border-radius:12px; padding:16px 18px; background: var(--background-fill-primary); }
.scorecard.flag { border-color: rgba(239,68,68,0.45); background: rgba(239,68,68,0.09); }
.scorecard.clear { border-color: rgba(34,197,94,0.45); background: rgba(34,197,94,0.09); }
.sc-name { font-size:1.02rem; font-weight:700; color: var(--body-text-color); }
.sc-sub { font-size:0.78rem; color: var(--body-text-color-subdued); margin-bottom:10px; }
.sc-prob { font-size:2.5rem; font-weight:700; line-height:1; color: var(--body-text-color);
  font-variant-numeric:tabular-nums; }
.sc-prob span { font-size:1.1rem; color: var(--body-text-color-subdued); margin-left:2px; }
.sc-bar { position:relative; height:8px; background: rgba(148,163,184,0.30);
  border-radius:4px; margin:12px 0 10px 0; }
.sc-fill { position:absolute; height:100%; border-radius:4px; background:#6366f1; }
.scorecard.flag .sc-fill { background:#ef4444; }
.scorecard.clear .sc-fill { background:#22c55e; }
.sc-thr { position:absolute; top:-3px; width:2px; height:14px; background: var(--body-text-color); }
.sc-verdict { font-size:0.82rem; font-weight:700; letter-spacing:0.06em; }
.scorecard.flag .sc-verdict { color:#ef4444; }
.scorecard.clear .sc-verdict { color:#22c55e; }
.sc-thrlab { font-size:0.72rem; color: var(--body-text-color-subdued); margin-top:3px; }

.banner { display:flex; align-items:center; gap:14px; padding:13px 18px; border-radius:10px;
  margin-top:4px; flex-wrap:wrap; }
.banner.good { background: rgba(34,197,94,0.10); border:1px solid rgba(34,197,94,0.40); }
.banner.bad { background: rgba(239,68,68,0.10); border:1px solid rgba(239,68,68,0.40); }
.banner.warn { background: rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.40); }
.b-label { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.07em;
  color: var(--body-text-color-subdued); }
.b-truth { font-weight:750; font-size:0.95rem; letter-spacing:0.03em;
  color: var(--body-text-color); }
.b-msg { color: var(--body-text-color); opacity:0.85; font-size:0.9rem; }

.tilehead { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.07em;
  color: var(--body-text-color-subdued); margin:2px 2px 7px 2px; }
.tilerow { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:10px; }
.tile { flex:1; min-width:150px; border:1px solid var(--border-color-primary);
  border-radius:10px; padding:13px 15px; background: var(--background-fill-primary); }
.t-label { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.07em;
  color: var(--body-text-color-subdued); }
.t-value { font-size:1.6rem; font-weight:700; color: var(--body-text-color);
  font-variant-numeric:tabular-nums; line-height:1.2; }
.t-sub { font-size:0.76rem; color: var(--body-text-color-subdued); }
.optnote { padding:12px 16px; background: rgba(99,102,241,0.12);
  border:1px solid rgba(99,102,241,0.40); border-radius:10px;
  color: var(--body-text-color); font-size:0.92rem; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
)

with gr.Blocks(title="Sequential Fraud Detection") as demo:
    gr.HTML(
        "<div id='hero'><h1>Sequential Fraud Detection</h1>"
        "<p>A GRU reads each card's last 10 transactions. A LightGBM baseline sees the same "
        "features without the sequence. Every score below is computed live.</p></div>"
    )

    with gr.Tabs():
        with gr.Tab("Score a transaction"):
            with gr.Row():
                scenario = gr.Dropdown(
                    choices=CFG["scenario_order"],
                    value=CFG["scenario_order"][0],
                    label="Pick a scenario",
                    scale=3,
                )
                shuffle = gr.Button("Load another case", variant="primary", scale=1)

            case_header = gr.HTML()
            history = gr.HTML()
            panels = gr.HTML()
            banner = gr.HTML()

            scenario.change(load_case, scenario, [case_header, history, panels, banner])
            shuffle.click(load_case, scenario, [case_header, history, panels, banner])
            demo.load(load_case, scenario, [case_header, history, panels, banner])

        with gr.Tab("Cost explorer"):
            gr.Markdown(
                "A missed fraud costs the full transaction amount. A false positive costs a "
                "manual review. Move the inputs and watch where the optimum goes: as reviews "
                "get more expensive, the threshold rises and you alert less."
            )
            with gr.Row():
                review_cost = gr.Slider(1, 50, value=CFG["default_review_cost"], step=0.5,
                                        label="Cost of one manual review ($)")
                threshold = gr.Slider(0.0, 1.0, value=THR_GRU, step=0.001,
                                      label="Decision threshold")
            snap = gr.Button("Snap to the cost minimising threshold")

            tiles = gr.HTML()
            cost_plot = gr.Plot(show_label=False)

            for control in (review_cost, threshold):
                control.change(explore, [review_cost, threshold], [tiles, cost_plot])
            snap.click(snap_to_optimal, review_cost, threshold)
            demo.load(explore, [review_cost, threshold], [tiles, cost_plot])

        with gr.Tab("How it works"):
            gr.Markdown(ABOUT)
            gr.Plot(pr_figure(), show_label=False)


if __name__ == "__main__":
    ids = META.sample_id.values[:25]
    d_gru = max(abs(score_gru(int(i)) - float(META.score_gru[i])) for i in ids)
    d_lgb = max(abs(score_lgb(int(i)) - float(META.score_lgb[i])) for i in ids)
    print(f"[check] max score drift  GRU {d_gru:.2e}  LightGBM {d_lgb:.2e}")

    demo.launch(theme=THEME, css=CSS)
