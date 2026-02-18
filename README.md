# Customer Churn Prediction

A production-ready machine learning pipeline to predict customer churn using classification models, with a clean modular architecture ready for GitHub and team collaboration.

## 🎯 Project Overview

Customer churn prediction helps businesses identify at-risk customers and implement retention strategies. This project implements a complete, end-to-end ML pipeline — from raw data ingestion to saved model artifacts — with structured logging, a CLI interface, and a reusable preprocessing pipeline.

**Key Features:**
- Modular `src/` package with clean separation of concerns
- Centralized config (`src/config.py`) — no hardcoded paths or hyperparameters
- Structured logging throughout the pipeline
- CLI interface via `argparse` for flexible execution
- ROC curves saved as PNG files (CI/script-safe — no `plt.show()`)
- Results saved to `results/` as both CSV and JSON
- Trained model pipelines saved to `models/` as `.joblib` files

## 📁 Project Structure

```
customer-churn-prediction/
├── data/
│   └── Churn_Modelling.csv          # Raw customer churn dataset
├── models/                           # Saved model pipelines (.joblib)
│   └── .gitkeep
├── notebooks/
│   ├── eda.ipynb                     # Exploratory Data Analysis
│   ├── feature_engineering.ipynb     # Feature engineering experiments
│   ├── models.ipynb                  # Baseline model training
│   ├── advancedmodels.ipynb          # XGBoost, LightGBM, CatBoost
│   └── preprocessing_pipeline.joblib # Saved preprocessing pipeline (generated)
├── results/                          # Generated evaluation outputs
│   ├── results.csv                   # Metrics comparison table
│   ├── results.json                  # Metrics in JSON format
│   ├── plots/                        # ROC curve PNGs
│   └── .gitkeep
├── src/
│   ├── __init__.py                   # Marks src as a Python package
│   ├── config.py                     # Centralized paths & hyperparameters
│   ├── data_loader.py                # Data ingestion & target detection
│   ├── preprocessing.py              # Scikit-learn preprocessing pipeline
│   ├── trainer.py                    # Model training & persistence
│   ├── evaluation.py                 # Metrics computation & ROC plots
│   └── results.py                    # Results saving utilities
├── main.py                           # CLI entry point — runs the full pipeline
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Excludes cache, models, .DS_Store, etc.
└── README.md
```

## 📊 Dataset

The project uses `Churn_Modelling.csv` containing 10,000 bank customer records:

| Feature | Type | Description |
|---------|------|-------------|
| CreditScore | Numeric | Customer credit score |
| Geography | Categorical | Country (France / Germany / Spain) |
| Gender | Categorical | Male / Female |
| Age | Numeric | Customer age |
| Tenure | Numeric | Years as a customer |
| Balance | Numeric | Account balance |
| NumOfProducts | Numeric | Number of bank products |
| HasCrCard | Numeric | Has credit card (0/1) |
| IsActiveMember | Numeric | Active member (0/1) |
| EstimatedSalary | Numeric | Estimated annual salary |
| **Exited** | **Target** | **Churned (1) / Retained (0)** |

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd customer-churn-prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Full Pipeline (CLI)

```bash
# Default — uses paths from src/config.py
python main.py

# Custom paths
python main.py \
  --data-path data/Churn_Modelling.csv \
  --models-dir models \
  --results-dir results \
  --plots-dir results/plots \
  --log-level INFO
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | `data/Churn_Modelling.csv` | Path to input CSV |
| `--models-dir` | `models/` | Directory to save model `.joblib` files |
| `--results-dir` | `results/` | Directory to save CSV/JSON results |
| `--plots-dir` | `results/plots/` | Directory to save ROC curve PNGs |
| `--log-level` | `INFO` | Logging verbosity (`DEBUG/INFO/WARNING/ERROR`) |

### Running Notebooks

```bash
jupyter notebook
```

Execute notebooks in order:
1. `eda.ipynb` — Explore data distributions and correlations
2. `feature_engineering.ipynb` — Engineer and transform features
3. `models.ipynb` — Train and evaluate baseline models
4. `advancedmodels.ipynb` — XGBoost, LightGBM, CatBoost

### Using Saved Artifacts

```python
import joblib

# Load a trained model pipeline
pipeline = joblib.load("models/logisticregression_pipeline.joblib")

# Predict on new data (raw, unprocessed DataFrame)
predictions = pipeline.predict(X_new)
probabilities = pipeline.predict_proba(X_new)[:, 1]

# Load just the preprocessing pipeline
preprocessor = joblib.load("notebooks/preprocessing_pipeline.joblib")
X_processed = preprocessor.transform(X_new)
```

## 🤖 Models

### Baseline Models

| Model | Strengths | Notes |
|-------|-----------|-------|
| **Logistic Regression** | Fast, interpretable, good baseline | `liblinear` solver, class-balanced |
| **Decision Tree** | Non-linear, feature importance | `max_depth=5` to prevent overfitting |

### Pipeline Architecture

Each model is wrapped in a full sklearn `Pipeline`:
```
Raw DataFrame → ColumnTransformer (scale numerics, encode categoricals) → Classifier
```

This means the saved `.joblib` file handles all preprocessing automatically — just call `.predict(X_raw)`.

## 📈 Evaluation Metrics

| Metric | Description | Priority |
|--------|-------------|----------|
| **Recall** | Fraction of actual churners caught | ⭐ Primary |
| **F1-Score** | Harmonic mean of precision & recall | High |
| **ROC-AUC** | Discrimination ability | High |
| **Precision** | Fraction of predicted churners that are real | Medium |
| **Accuracy** | Overall correctness | Low (misleading with imbalance) |

> **Why Recall?** Missing a churner (false negative) is more costly than a false alarm. Recall is prioritized to maximize retention campaign coverage.

## 📊 Outputs

After running `python main.py`:

```
results/
├── results.csv          # Model comparison table
├── results.json         # Same data in JSON format
└── plots/
    ├── roc_logisticregression.png
    └── roc_decisiontree.png

models/
├── logisticregression_pipeline.joblib
└── decisiontree_pipeline.joblib
```

## 🔮 Next Steps

- [ ] Add cross-validation (StratifiedKFold)
- [ ] Hyperparameter tuning (GridSearchCV / Optuna)
- [ ] Advanced models: XGBoost, LightGBM, CatBoost (`advancedmodels.ipynb`)
- [ ] SHAP values for model explainability
- [ ] SMOTE / class-weight tuning for imbalance
- [ ] REST API with FastAPI for model serving
- [ ] Unit tests for `src/` modules
- [ ] GitHub Actions CI workflow

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Commit your changes: `git commit -am 'Add improvement'`
4. Push: `git push origin feature/my-improvement`
5. Open a Pull Request

## 📝 License

MIT License — see `LICENSE` for details.

## 👤 Author

**Saumya Jain**
>>>>>>> e42f58a (adding more models)
