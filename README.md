# Customer Churn Prediction

A machine learning project to predict customer churn using baseline classification models (Logistic Regression and Decision Tree) with a complete end-to-end pipeline from data exploration to model evaluation.

## 🎯 Project Overview

Customer churn prediction is critical for businesses to identify at-risk customers and implement retention strategies. This project implements a complete machine learning pipeline to predict whether a customer will churn based on their demographic and behavioral features.

**Key Features:**
- Comprehensive exploratory data analysis (EDA)
- Feature engineering and preprocessing pipeline
- Baseline model training and evaluation
- Automated results storage and visualization
- Reusable preprocessing pipeline for production deployment

## 📁 Project Structure

```
├── data/
│   └── Churn_Modelling.csv          # Raw customer churn dataset
├── notebooks/
│   ├── eda.ipynb                     # Exploratory Data Analysis
│   ├── feature_engineering.ipynb     # Feature engineering pipeline
│   ├── models.ipynb                  # Baseline model training & evaluation
│   └── preprocessing_pipeline.joblib # Saved preprocessing pipeline
├── src/
│   ├── evaluation.py                 # Model evaluation utilities
│   ├── results.py                    # Results saving utilities
│   └── __pycache__/                  # Python cache files
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies (recommended)
```

## 📊 Dataset

The project uses the `Churn_Modelling.csv` dataset containing customer information including:
- **Demographics:** Age, Gender, Geography
- **Banking Information:** Credit Score, Balance, Number of Products
- **Customer Behavior:** Active Member status, Tenure, Estimated Salary
- **Target Variable:** Exited (1 = Churned, 0 = Retained)

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Jupyter Notebook

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-churn-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib jupyter
```

Or create a `requirements.txt` and install:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Complete Pipeline

1. **Start Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Execute notebooks in sequence:**
   - `eda.ipynb` - Understand data patterns and distributions
   - `feature_engineering.ipynb` - Create and transform features
   - `models.ipynb` - Train and evaluate models

3. **Run all cells in models.ipynb:**
   - Option 1: Use `Kernel → Restart & Run All`
   - Option 2: Execute cells individually for step-by-step analysis

### Accessing Results

After running the models notebook:
- **Metrics:** Check `src/results.csv` and `src/results.json`
- **Visualizations:** ROC curves displayed inline in the notebook
- **Preprocessing Pipeline:** Saved as `notebooks/preprocessing_pipeline.joblib`

### Using the Saved Pipeline

```python
import joblib

# Load the preprocessing pipeline
pipeline = joblib.load('notebooks/preprocessing_pipeline.joblib')

# Apply to new data
X_new_processed = pipeline.transform(X_new)
```

## 🤖 Models

### Baseline Models

**1. Logistic Regression**
- Linear classification model
- Good interpretability
- Fast training and prediction
- Suitable for baseline performance

**2. Decision Tree Classifier**
- Non-linear decision boundaries
- Feature importance insights
- Prone to overfitting (baseline version)
- Interpretable decision rules

### Model Training Process

1. Data preprocessing (scaling, encoding)
2. Train-test split (80-20)
3. Model fitting on training data
4. Evaluation on test set
5. Results storage and visualization

## 📈 Evaluation Metrics

Models are evaluated using multiple metrics to provide comprehensive performance assessment:

| Metric | Description | Importance |
|--------|-------------|------------|
| **Accuracy** | Overall correctness of predictions | General performance indicator |
| **Precision** | True positives among predicted positives | Minimizes false alarms |
| **Recall** | True positives among actual positives | **Primary focus** - identifies actual churners |
| **F1-Score** | Harmonic mean of precision and recall | Balanced metric |
| **ROC-AUC** | Area under the ROC curve | Discrimination ability |

**Focus Metric:** Recall is prioritized because identifying customers at risk of churning (true positives) is more valuable than avoiding false positives in retention campaigns.

## 📊 Results

Results are automatically saved in two formats:

1. **CSV Format** (`src/results.csv`):
   - Tabular format for easy comparison
   - Includes all metrics for both models

2. **JSON Format** (`src/results.json`):
   - Structured format for programmatic access
   - Suitable for integration with dashboards

**Visualization:**
- ROC curves comparing both models
- Performance comparison charts (when available)

## 🔮 Next Steps

### Immediate Improvements
- [ ] Implement ensemble methods (Random Forest, XGBoost, LightGBM)
- [ ] Add cross-validation for robust performance estimates
- [ ] Perform hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- [ ] Handle class imbalance (SMOTE, class weights)

### Advanced Analytics
- [ ] Feature importance analysis and selection
- [ ] Model explainability (SHAP values, LIME)
- [ ] Customer segmentation analysis
- [ ] Cost-benefit analysis for retention strategies

### Production Readiness
- [ ] Create model serving API (Flask/FastAPI)
- [ ] Implement model monitoring and versioning
- [ ] Add automated retraining pipeline
- [ ] Deploy to cloud platform (AWS/GCP/Azure)

### Documentation & Reporting
- [ ] Create executive summary report
- [ ] Build interactive dashboard for stakeholders
- [ ] Document model limitations and assumptions
- [ ] Add unit tests for preprocessing and evaluation code

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 👥 Authors
SAUMYA JAIN
