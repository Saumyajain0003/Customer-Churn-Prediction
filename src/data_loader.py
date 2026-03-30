import sqlite3
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.config import DB_PATH, FEATURE_QUERY_PATH

logger = logging.getLogger(__name__)

# Columns that are unique identifiers and add no predictive value
_DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

# Possible names for the target column
_TARGET_CANDIDATES = ["Exited", "Churn", "churn", "is_churn", "IsChurn"]


def load_data(filepath: str | Path = None) -> pd.DataFrame:
    """
    Load data by executing the feature engineering SQL query against the SQLite database.

    Args:
        filepath: Legacy argument (kept for compatibility, but ignored in favor of DB_PATH).

    Returns:
        Cleaned DataFrame containing engineered features from the database.

    Raises:
        FileNotFoundError: If the DB or SQL query file is missing.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at: {DB_PATH}. Did you run seed_db.py?")
        
    if not FEATURE_QUERY_PATH.exists():
        raise FileNotFoundError(f"SQL query not found at: {FEATURE_QUERY_PATH}")

    logger.info(f"Connecting to SQLite DB at {DB_PATH.resolve()}")
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Read the advanced feature engineering query
        with open(FEATURE_QUERY_PATH, 'r') as file:
            query = file.read()
            
        logger.info("Executing SQL feature engineering query...")
        df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns from database")
    finally:
        conn.close()

    # Drop unnamed leading index column if present
    if df.columns[0] == "" or str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(df.columns[0], axis=1)

    # Drop identifier columns that exist in the dataframe
    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped identifier columns: {cols_to_drop}")

    return df


def get_target(df: pd.DataFrame):
    """
    Auto-detect and extract the target column from the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (X, y, target_col_name).

    Raises:
        KeyError: If no known target column is found.
    """
    target_col = next((c for c in _TARGET_CANDIDATES if c in df.columns), None)

    if target_col is None:
        raise KeyError(
            f"Could not find a target column. Expected one of: {_TARGET_CANDIDATES}. "
            f"Got columns: {list(df.columns)}"
        )

    logger.info(f"Target column detected: '{target_col}'")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Log class distribution
    dist = y.value_counts(normalize=True).to_dict()
    logger.info(f"Class distribution: { {k: f'{v:.1%}' for k, v in dist.items()} }")

    return X, y, target_col
