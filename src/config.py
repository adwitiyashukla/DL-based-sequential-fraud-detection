"""
Shared paths and hyperparameters.

All paths are derived from the location of this file, so the repository can be
cloned anywhere. No absolute paths are hardcoded.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
SCREENSHOT_DIR = RESULTS_DIR / "screenshots"

TRAIN_CSV = DATA_DIR / "fraudTrain.csv"
TEST_CSV = DATA_DIR / "fraudTest.csv"

FEATURE_CACHE = CACHE_DIR / "features.npz"
GRU_CHECKPOINT = CACHE_DIR / "gru.pth"
LGB_MODEL = CACHE_DIR / "lightgbm.txt"

# Score files, written by the training scripts and read by evaluate.py.
SCORES = {
    "gru": CACHE_DIR / "scores_gru.npy",
    "lightgbm": CACHE_DIR / "scores_lightgbm.npy",
    "logreg": CACHE_DIR / "scores_logreg.npy",
}

MODEL_LABELS = {
    "gru": "GRU (sequence)",
    "lightgbm": "LightGBM (no sequence)",
    "logreg": "Logistic regression",
}

# Sequence configuration.
SEQ_LEN = 10

# Model configuration.
CAT_EMB_DIM = 16
NUM_PROJ_DIM = 32
GRU_HIDDEN = 64
DROPOUT = 0.2

# Training configuration.
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 4
NUM_THREADS = 10
SEED = 42

# Cost model used for threshold selection.
# A missed fraud is assumed to cost the full transaction amount.
# A false positive is assumed to cost a fixed manual review expense.
REVIEW_COST = 5.0

# Alert budgets, expressed as the fraction of transactions a review team can look at.
ALERT_BUDGETS = (0.001, 0.01)


def ensure_dirs() -> None:
    """Create the output directories if they do not already exist."""
    for d in (CACHE_DIR, RESULTS_DIR, SCREENSHOT_DIR):
        d.mkdir(parents=True, exist_ok=True)
