"""
config.py — Centralized configuration for the Customer Churn Prediction pipeline.
All paths and hyperparameters are defined here to avoid hardcoding in scripts.
"""
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_FILE = DATA_DIR / "Churn_Modelling.csv"

# ── Train / Test Split ────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5  # Number of folds for k-fold cross-validation

# ── Model Hyperparameters ─────────────────────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS = {
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "max_iter": 1000,
}

DECISION_TREE_PARAMS = {
    "class_weight": "balanced",
    "max_depth": 5,
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}

# ── Results ───────────────────────────────────────────────────────────────────
RESULTS_CSV = RESULTS_DIR / "results.csv"
RESULTS_JSON = RESULTS_DIR / "results.json"

# ── Preprocessing pipeline (for notebook reuse) ───────────────────────────────
PREPROCESSING_PIPELINE_PATH = NOTEBOOKS_DIR / "preprocessing_pipeline.joblib"
