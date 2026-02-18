import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Columns that are unique identifiers and add no predictive value
_DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

# Possible names for the target column
_TARGET_CANDIDATES = ["Exited", "Churn", "churn", "is_churn", "IsChurn"]


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load and clean the churn dataset from a CSV file.

    Steps:
    - Drops unique identifier columns (RowNumber, CustomerId, Surname)
    - Drops any leading unnamed index columns

    Args:
        filepath: Path to the CSV file.

    Returns:
        Cleaned DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path.resolve()}")

    logger.info(f"Loading data from {path.resolve()}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

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
