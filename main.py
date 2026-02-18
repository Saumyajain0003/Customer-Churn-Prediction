"""
main.py — End-to-end Customer Churn Prediction Pipeline

Usage:
    python main.py
    python main.py --data-path data/Churn_Modelling.csv
    python main.py --data-path data/Churn_Modelling.csv --models-dir models --results-dir results
"""

import argparse
import logging
import sys
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    DATA_FILE,
    MODELS_DIR,
    RESULTS_DIR,
    PREPROCESSING_PIPELINE_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    LOGISTIC_REGRESSION_PARAMS,
    DECISION_TREE_PARAMS,
)
from src.data_loader import load_data, get_target
from src.preprocessing import get_preprocessing_pipeline
from src.trainer import train_and_evaluate, save_model
from src.results import save_results


# ── Logging setup ─────────────────────────────────────────────────────────────
def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction — End-to-End Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_FILE,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory to save trained model pipelines.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to save evaluation results (CSV/JSON).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory to save ROC curve plots. Defaults to <results-dir>/plots.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(
    data_path: Path,
    models_dir: Path,
    results_dir: Path,
    plots_dir: Path,
) -> None:
    logger = logging.getLogger(__name__)

    # 1. Load Data
    logger.info("Step 1/7 — Loading data")
    df = load_data(data_path)
    X, y, target_col = get_target(df)

    # 2. Identify Column Types
    logger.info("Step 2/7 — Identifying feature types")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info(f"  Numeric features ({len(numeric_cols)}):     {numeric_cols}")
    logger.info(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # 3. Train / Test Split
    logger.info("Step 3/7 — Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # 4. Preprocessing Pipeline
    logger.info("Step 4/7 — Building preprocessing pipeline")
    preprocessor = get_preprocessing_pipeline(numeric_cols, categorical_cols)

    # 5. Define Models
    logger.info("Step 5/7 — Defining models")
    models_to_train = {
        "LogisticRegression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        "DecisionTree":       DecisionTreeClassifier(**DECISION_TREE_PARAMS),
    }

    # 6. Train and Evaluate
    logger.info("Step 6/7 — Training and evaluating models")
    overall_results = {}
    for name, model in models_to_train.items():
        pipeline, metrics = train_and_evaluate(
            name, model, preprocessor,
            X_train, y_train, X_test, y_test,
            plots_dir=plots_dir,
        )
        overall_results[name] = metrics
        save_model(pipeline, models_dir / f"{name.lower()}_pipeline.joblib")

    # 7. Save Results
    logger.info("Step 7/7 — Saving results")
    save_results(overall_results, out_dir=results_dir, filename="results.csv")

    # Save standalone preprocessor for notebook reuse
    preprocessor.fit(X_train)
    PREPROCESSING_PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSING_PIPELINE_PATH)
    logger.info(f"Preprocessing pipeline saved to {PREPROCESSING_PIPELINE_PATH}")

    logger.info("Pipeline complete ✓")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = _parse_args()
    _setup_logging(args.log_level)

    plots_dir = args.plots_dir or (args.results_dir / "plots")

    run_pipeline(
        data_path=args.data_path,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        plots_dir=plots_dir,
    )
