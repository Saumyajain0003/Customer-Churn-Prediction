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
import sqlite3
import pandas as pd
from pathlib import Path

# Need to append ROOT_DIR to path if calling predict.py from another directory
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import DB_PATH

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
        help="Path to save predictions (default: print to console and save to DB)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def predict(model_path: Path, data_path: Path, output_path: Path = None) -> None:
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

    # Save customer IDs before dropping them for prediction, if they exist
    customer_ids = df['CustomerId'].tolist() if 'CustomerId' in df.columns else [None]*len(df)
    
    # Drop identifier columns safely before predicting
    cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
    df_for_prediction = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(df_for_prediction)
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df_for_prediction)[:, 1]
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(df_for_prediction)

    # Create results dataframe
    results_df = pd.DataFrame({
        "customer_id": customer_ids,
        "prediction": predictions,
    })
    
    if probabilities is not None:
        results_df["probability"] = probabilities
    else:
        results_df["probability"] = [1.0 if p == 1 else 0.0 for p in predictions]
    
    logger.info(f"✓ Predictions complete")
    logger.info(f"  Churn (1): {(predictions == 1).sum()} customers")
    logger.info(f"  No Churn (0): {(predictions == 0).sum()} customers")

    # Save to SQLite Database
    if DB_PATH.exists():
        logger.info(f"Writing predictions back to SQLite Database at {DB_PATH.name}...")
        try:
            conn = sqlite3.connect(DB_PATH)
            write_df = results_df.copy()
            # If CustomerId was missing, we can't properly insert into our relational schema (foreign key).
            # So we only insert rows that have a valid customer_id.
            write_df = write_df.dropna(subset=['customer_id'])
            
            if len(write_df) > 0:
                # Rename to match our schema: customer_id, churn_probability
                db_write_df = write_df[['customer_id', 'probability']].rename(
                    columns={"probability": "churn_probability"}
                )
                
                db_write_df.to_sql("customer_churn_predictions", conn, if_exists="append", index=False)
                logger.info(f"✓ Wrote {len(db_write_df)} predictions to `customer_churn_predictions` table.")
            else:
                logger.warning("No valid Customer IDs found. Skipping database write-back.")
        except Exception as e:
            logger.error(f"Failed to write to database: {e}")
        finally:
            conn.close()
    else:
        logger.warning("Database not found. Skipping SQL write-back.")

    # Save or display results CSV
    if output_path:
        results_df.to_csv(output_path, index=False)
        logger.info(f"✓ Results CSV saved to {output_path}")
    else:
        print("\n" + "="*60)
        print("PREDICTIONS SUMMARY (First 5)")
        print("="*60)
        print(results_df.head(5).to_string(index=False))
        if len(results_df) > 5:
            print(f"... and {len(results_df)-5} more rows.")
        print("="*60 + "\n")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    predict(args.model, args.data, args.output)
