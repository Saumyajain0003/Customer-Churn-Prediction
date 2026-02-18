import logging
from pathlib import Path
from sklearn.pipeline import Pipeline
from src.evaluation import evaluate_model
import joblib

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
