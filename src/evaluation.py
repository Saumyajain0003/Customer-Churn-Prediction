import logging
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for scripts and CI
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    name: str,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    plots_dir: str | Path | None = None,
) -> dict:
    """
    Fit a model/pipeline, compute standard metrics, and optionally save a ROC curve.

    Args:
        name:      Human-readable model name (used in logs and plot titles).
        model:     A fitted or unfitted sklearn estimator or Pipeline.
        X_train:   Training features.
        y_train:   Training labels.
        X_test:    Test features.
        y_test:    Test labels.
        plots_dir: If provided, ROC curve PNG is saved here instead of displayed.

    Returns:
        Dictionary of metric name → value.
    """
    if not hasattr(model, "fit"):
        raise ValueError(f"Object '{name}' does not have a .fit() method.")

    logger.info(f"Fitting model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probability scores for ROC-AUC
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   None,
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception as exc:
            logger.warning(f"Could not compute ROC-AUC for {name}: {exc}")

    # ── Logging ───────────────────────────────────────────────────────────────
    logger.info(f"=== {name} Results ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {f'{v:.4f}' if v is not None else 'N/A'}")

    # Print classification report to console
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    if y_proba is not None and metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label=f'{name} (AUC={metrics["roc_auc"]:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {name}")
        ax.legend()
        fig.tight_layout()

        if plots_dir is not None:
            plots_path = Path(plots_dir)
            plots_path.mkdir(parents=True, exist_ok=True)
            out = plots_path / f"roc_{name.lower().replace(' ', '_')}.png"
            fig.savefig(out, dpi=150)
            logger.info(f"ROC curve saved to {out}")
        else:
            plt.show()

        plt.close(fig)

    return metrics
