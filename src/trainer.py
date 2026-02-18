import logging
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.evaluation import evaluate_model
import joblib
import numpy as np

logger = logging.getLogger(__name__)


def train_and_evaluate(
    name: str,
    model,
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test,
    plots_dir: str | Path | None = None,
) -> tuple[Pipeline, dict]:
    """
    Wrap a preprocessor and model into a Pipeline, train it, and evaluate.

    Args:
        name:        Human-readable model name.
        model:       Unfitted sklearn estimator.
        preprocessor: Fitted or unfitted sklearn Pipeline/ColumnTransformer.
        X_train, y_train: Training data.
        X_test, y_test:   Test data.
        plots_dir:   Directory to save ROC curve plots (None = show interactively).

    Returns:
        Tuple of (fitted Pipeline, metrics dict).
    """
    logger.info(f"Building pipeline for: {name}")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", model),
        ]
    )

    print(f"\n--- Training {name} ---")
    metrics = evaluate_model(
        name, pipeline, X_train, y_train, X_test, y_test, plots_dir=plots_dir
    )

    return pipeline, metrics


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    """
    Persist an entire pipeline (preprocessor + model) to disk.

    Args:
        pipeline: Fitted sklearn Pipeline.
        path:     Output file path (e.g. 'models/logisticregression_pipeline.joblib').
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
    logger.info(f"Model pipeline saved to {out.resolve()}")
    print(f"  ✓ Saved: {out}")


def train_and_evaluate_cv(
    name: str,
    model,
    preprocessor,
    X,
    y,
    cv_folds: int = 5,
) -> tuple[Pipeline, dict]:
    """
    Train and evaluate a model using k-fold cross-validation.

    Args:
        name:         Human-readable model name.
        model:        Unfitted sklearn estimator.
        preprocessor: Fitted or unfitted sklearn Pipeline/ColumnTransformer.
        X, y:         Full dataset (features and labels).
        cv_folds:     Number of folds for cross-validation (default: 5).

    Returns:
        Tuple of (fitted Pipeline on full data, average metrics dict across folds).
    """
    logger.info(f"Building pipeline for: {name}")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", model),
        ]
    )

    print(f"\n--- Training {name} (Cross-Validation with {cv_folds} folds) ---")

    # Define scorers for cross-validation
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
    }

    # Perform cross-validation with stratified k-fold
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv_splitter,
        scoring=scoring,
        return_train_score=True,
    )

    # Compute average metrics across all folds
    metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        test_scores = cv_results[f'test_{metric}']
        avg_score = test_scores.mean()
        std_score = test_scores.std()
        metrics[metric] = avg_score

        logger.info(f"  {metric}: {avg_score:.4f} (+/- {std_score:.4f})")
        print(f"  {metric}: {avg_score:.4f} (±{std_score:.4f})")

    # Train final model on full data for later use
    logger.info(f"Training final {name} model on full data")
    pipeline.fit(X, y)

    return pipeline, metrics
