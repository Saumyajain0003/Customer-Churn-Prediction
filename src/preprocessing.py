import logging
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def get_preprocessing_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """
    Build a robust scikit-learn preprocessing pipeline.

    Numeric features:
        - Median imputation (robust to outliers)
        - Standard scaling

    Categorical features:
        - Most-frequent imputation
        - One-hot encoding (unknown categories handled gracefully)

    Args:
        numeric_cols:     List of numeric feature column names.
        categorical_cols: List of categorical feature column names.

    Returns:
        A fitted-ready sklearn Pipeline with a ColumnTransformer step.
    """
    logger.info(
        f"Building preprocessing pipeline — "
        f"{len(numeric_cols)} numeric, {len(categorical_cols)} categorical features"
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(steps=[("preprocessor", preprocessor)])
