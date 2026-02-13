import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report


def evaluate_model(name, model, X_train, y_train, X_test, y_test, show_roc=True):
    """Fit model, compute standard metrics and optionally plot ROC. Returns metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except Exception:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None

    print(f'== {name} ==')
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: None")
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, digits=4))

    if show_roc and y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'{name} (AUC={metrics.get("roc_auc"):.3f})')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return metrics
