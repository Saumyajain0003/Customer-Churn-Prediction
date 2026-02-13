# Customer Churn Prediction

A machine learning project to predict customer churn using baseline models (Logistic Regression and Decision Tree).

## Project Structure
├── data/
│ └── Churn_Modelling.csv # Raw customer churn dataset
├── notebooks/
│ ├── eda.ipynb # Exploratory Data Analysis
│ ├── feature_engineering.ipynb # Feature engineering pipeline
│ ├── models.ipynb # Baseline model training & evaluation
│ └── preprocessing_pipeline.joblib # Saved preprocessing pipeline
├── src/
│ ├── evaluation.py # Model evaluation utilities
│ ├── results.py # Results saving utilities
│ └── pycache/
├── README.md # Project documentation

## Overview

This project trains and evaluates baseline classification models to predict whether a customer will churn. The pipeline includes:

- **Data Preprocessing:** Feature scaling, encoding, and transformation
- **Model Training:** Logistic Regression & Decision Tree classifiers
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC metrics
- **Results Storage:** Model metrics saved to CSV and JSON formats

## Requirements

- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib jupyter

Usage
Start Jupyter from project root:

Run the notebook sequence:

Open models.ipynb
Use Kernel → Restart & Run All to execute all cells
Check results:

Results saved to src/results.csv and src/results.json
ROC curves displayed inline in the notebook
Metrics
Models are evaluated on:

Accuracy: Overall correctness
Precision: True positives among predicted positives
Recall: True positives among actual positives (focus metric)
F1-Score: Harmonic mean of precision and recall
ROC-AUC: Area under the ROC curve

Next Steps
 Add advanced models (Random Forest, Gradient Boosting, etc.)
 Hyperparameter tuning
 Feature importance analysis
 Model comparison and selection
 Deployment pipeline
License
MIT