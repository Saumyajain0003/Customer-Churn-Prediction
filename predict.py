"""
predict.py — Simple script to make predictions on new data using trained models.

Usage:
    python predict.py --model models/randomforest_pipeline.joblib --data new_customers.csv
    python predict.py --model models/xgboost_pipeline.joblib --data new_customers.csv --output predictions.csv
"""

import argparse
import logging
import sys
import joblib
import pandas as pd
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make predictions using a trained model pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.joblib file)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to CSV file with new data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save predictions (default: print to console)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def predict(model_path: Path, data_path: Path, output_path: Path | None = None) -> None:
    logger = logging.getLogger(__name__)

    # Load model
    logger.info(f"Loading model from {model_path}")
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    model = joblib.load(model_path)
    logger.info("✓ Model loaded successfully")

    # Load data
    logger.info(f"Loading data from {data_path}")
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")

    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(df)
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df)[:, 1]
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(df)

    # Create results dataframe
    results_df = pd.DataFrame({
        "prediction": predictions,
    })
    
    if probabilities is not None:
        results_df["probability"] = probabilities
    
    logger.info(f"✓ Predictions complete")
    logger.info(f"  Churn (1): {(predictions == 1).sum()} customers")
    logger.info(f"  No Churn (0): {(predictions == 0).sum()} customers")

    # Save or display results
    if output_path:
        results_df.to_csv(output_path, index=False)
        logger.info(f"✓ Results saved to {output_path}")
    else:
        print("\n" + "="*60)
        print("PREDICTIONS")
        print("="*60)
        print(results_df.to_string(index=False))
        print("="*60 + "\n")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    predict(args.model, args.data, args.output)
