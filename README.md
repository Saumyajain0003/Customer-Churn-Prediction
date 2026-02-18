# Customer Churn Prediction 

Predict which customers will churn . This project trains both baseline and advanced models, saves them, and generates results automatically.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python main.py
```

This will:
- Load the data
- Preprocess it (scaling, encoding)
- Train 4 models:
  - **Logistic Regression** (baseline - fast & interpretable)
  - **Decision Tree** (baseline - easy to understand)
  - **Random Forest** (advanced - ensemble method)
  - **XGBoost** (advanced - powerful gradient boosting)
- Save trained models to `models/`
- Save results to `results/results.csv`
- Save ROC curves to `results/plots/`

### 3. View Results
Check `results/results.csv` for model performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).

## Project Structure

```
.
├── data/
│   └── Churn_Modelling.csv          ← Your raw data
├── models/                           ← Saved trained models (.joblib)
├── results/                          ← Performance metrics & plots
├── src/
│   ├── config.py                    ← Configuration & hyperparameters
│   ├── data_loader.py               ← Load & prepare data
│   ├── preprocessing.py             ← Data preprocessing pipeline
│   ├── trainer.py                   ← Train & save models
│   ├── evaluation.py                ← Calculate metrics & ROC curves
│   └── results.py                   ← Save results
├── main.py                          ← Run the full pipeline
└── requirements.txt                 ← Python packages
```

## CLI Options

```bash
# Use custom data path
python main.py --data-path path/to/your/data.csv

# Specify output directories
python main.py --models-dir ./my_models --results-dir ./my_results

# Adjust logging verbosity
python main.py --log-level DEBUG
```

## Making Predictions

To predict on new data:

```python
import joblib
import pandas as pd

# Load a trained model
model = joblib.load('models/randomforest_pipeline.joblib')

# Load new data (must have same features as training data)
new_data = pd.read_csv('new_customers.csv')

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]  # Churn probability
```

## Model Explanations

| Model | Type | When to Use | Pros | Cons |
|-------|------|-------------|------|------|
| **Logistic Regression** | Baseline | Quick baseline | Fast, interpretable, stable | Lower accuracy |
| **Decision Tree** | Baseline | Interpretability | Very explainable, no scaling needed | Overfitting risk |
| **Random Forest** | Advanced | Good default | High accuracy, robust | Less interpretable |
| **XGBoost** | Advanced | Complex data | Highest accuracy, fast | More complex tuning |

## Hyperparameters

Edit `src/config.py` to change:
- Model hyperparameters (learning rates, tree depth, etc.)
- Train/test split ratio (default: 80/20)
- Random seed (for reproducibility)

## Output Files

- `results/results.csv` - Model metrics comparison
- `results/plots/roc_*.png` - ROC curves for each model
- `models/*_pipeline.joblib` - Saved trained models (ready for predictions)

## Troubleshooting

**"Module not found" error?**
```bash
pip install -r requirements.txt
```

**"No module named 'xgboost'"?**
```bash
pip install xgboost  # Optional - XGBoost will be skipped if not installed
```

**Want to change features or preprocessing?**
Edit `src/preprocessing.py` to modify the pipeline.

## For Your Project Report

- **Models Used**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Dataset**: Customer churn data with customer features
- **Approach**: Standard ML pipeline with train/test split (80/20)
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Best Model**: Check `results/results.csv` for performance

---

### AUTHOR:
SAUMYA JAIN
