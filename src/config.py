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

SEQ_LEN = 10

CAT_EMB_DIM = 16
NUM_PROJ_DIM = 32
GRU_HIDDEN = 64
DROPOUT = 0.2

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 4
NUM_THREADS = 10
SEED = 42

REVIEW_COST = 5.0

ALERT_BUDGETS = (0.001, 0.01)


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RESULTS_DIR, SCREENSHOT_DIR):
        d.mkdir(parents=True, exist_ok=True)
