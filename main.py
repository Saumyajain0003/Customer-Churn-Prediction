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
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.config import (
    DATA_FILE,
    MODELS_DIR,
    RESULTS_DIR,
    PREPROCESSING_PIPELINE_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    CROSS_VALIDATION_FOLDS,
    LOGISTIC_REGRESSION_PARAMS,
    DECISION_TREE_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)
from src.data_loader import load_data, get_target
from src.preprocessing import get_preprocessing_pipeline
from src.trainer import train_and_evaluate_cv, save_model
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
    logger.info("Step 2/5 — Identifying feature types")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info(f"  Numeric features ({len(numeric_cols)}):     {numeric_cols}")
    logger.info(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # 3. Preprocessing Pipeline
    logger.info("Step 3/5 — Building preprocessing pipeline")
    preprocessor = get_preprocessing_pipeline(numeric_cols, categorical_cols)

    # 4. Define Models
    logger.info("Step 4/5 — Defining models")
    models_to_train = {
        "LogisticRegression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        "DecisionTree":       DecisionTreeClassifier(**DECISION_TREE_PARAMS),
        "RandomForest":       RandomForestClassifier(**RANDOM_FOREST_PARAMS),
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models_to_train["XGBoost"] = XGBClassifier(**XGBOOST_PARAMS)
        logger.info("  ✓ XGBoost is available")
    else:
        logger.info("  ⚠ XGBoost not installed (skipping)")

    # 5. Train and Evaluate with Cross-Validation
    logger.info(f"Step 5/5 — Training with {CROSS_VALIDATION_FOLDS}-fold cross-validation")
    overall_results = {}
    for name, model in models_to_train.items():
        pipeline, metrics = train_and_evaluate_cv(
            name, model, preprocessor,
            X, y,  # Use full dataset for CV
            cv_folds=CROSS_VALIDATION_FOLDS,
        )
        overall_results[name] = metrics
        save_model(pipeline, models_dir / f"{name.lower()}_pipeline.joblib")

    # Find and display best model
    print("\n" + "="*60)
    print("MODEL COMPARISON (Cross-Validation Results)")
    print("="*60)
    results_df = __import__('pandas').DataFrame(overall_results).T
    results_df = results_df.round(4)
    print(results_df.to_string())
    
    best_model = results_df['f1'].idxmax()
    best_f1 = results_df.loc[best_model, 'f1']
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model} (F1-Score: {best_f1:.4f})")
    print("="*60 + "\n")
    logger.info(f"Best model: {best_model} (F1-Score: {best_f1:.4f})")

    # Save Results
    logger.info("Saving results")
    save_results(overall_results, out_dir=results_dir, filename="results_cv.csv")

    # Save standalone preprocessor for notebook reuse (fit on full data)
    preprocessor.fit(X)
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
