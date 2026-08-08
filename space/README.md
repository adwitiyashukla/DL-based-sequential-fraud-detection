---
title: Sequential Fraud Detection
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: true
license: mit
short_description: A GRU scoring card transaction sequences, live
tags:
  - fraud-detection
  - pytorch
  - deep-learning
  - fintech
  - gru
---

# Sequential Fraud Detection

Most credit card fraud models score each transaction in isolation. Fraud is a behavioural
signal: what matters is that *this card* has never behaved this way before.

This demo runs a GRU over each card's last 10 transactions and compares it against a
LightGBM model given the identical features without sequence context. On a held out time
period of 555,719 transactions the GRU reaches **0.965 PR-AUC** against **0.899**, and at the
cost minimising threshold it misses **31** of 2,145 frauds where LightGBM misses **112**.

**Every score in the first tab is a real PyTorch forward pass**, computed when you click.
Nothing is looked up.

## The two tabs

**Score a transaction.** Pick a scenario, including the interesting one where the sequence
model catches fraud the flat model misses, and see the card's recent history alongside both
models' verdicts.

**Cost explorer.** A missed fraud costs the transaction amount; a false positive costs a
manual review. Move the review cost and watch the optimal threshold move with it. The
decision threshold is a business parameter, not a modelling constant.

## Limits worth stating

The data is simulated (Sparkov via Kaggle). Rule generated fraud is far more learnable than
the adversarial kind, so 0.965 PR-AUC is not a production number; the relative comparison
between models is the meaningful output. The cost minimising threshold is also selected on
the test set, which makes the dollar figures optimistic.

Trained on CPU: 22,577 parameters, 4 epochs, 7.1 minutes on a laptop with no GPU.

[Full code and methodology on GitHub](https://github.com/adwitiyashukla/DL-based-sequential-fraud-detection)
