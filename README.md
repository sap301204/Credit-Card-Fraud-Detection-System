
# Credit Card Fraud Detection System

## Overview

This project is a machine learning-based fraud detection system that predicts whether a credit card transaction is fraudulent or genuine.

It uses synthetic transaction data, handles class imbalance using SMOTE, trains a Random Forest model, and provides a Streamlit dashboard for fraud risk prediction.

## Problem Statement

Credit card fraud causes financial loss to banks, fintech companies, payment gateways, and customers. Since fraud transactions are rare compared to genuine transactions, this becomes an imbalanced classification problem.

The goal of this project is to detect suspicious transactions and classify them into:

- ALLOW
- REVIEW
- BLOCK

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- SMOTE
- Random Forest
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Project Workflow

Transaction Data
        ↓
Data Preprocessing
        ↓
SMOTE Imbalance Handling
        ↓
Random Forest Model Training
        ↓
Fraud Probability Prediction
        ↓
ALLOW / REVIEW / BLOCK Decision
        ↓
Streamlit Dashboard

## Features

- Synthetic transaction data generation
- Fraud and non-fraud classification
- Class imbalance handling using SMOTE
- Random Forest model training
- Precision, recall, F1-score, and PR-AUC evaluation
- Confusion matrix visualization
- Precision-recall curve
- Feature importance chart
- Streamlit fraud risk dashboard

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run Streamlit app

streamlit run app.py

## Model Evaluation

The model is evaluated using:

- Precision
- Recall
- F1-score
- PR-AUC
- Confusion Matrix
- Precision-Recall Curve

Accuracy alone is not used as the main metric because fraud detection is an imbalanced classification problem.

## Decision Logic

| Fraud Probability | Decision |
|---|---|
| Below 40% | ALLOW |
| 40% to 70% | REVIEW |
| Above 70% | BLOCK |

## Business Impact

This project simulates how banks and fintech companies detect risky transactions, reduce fraud losses, and support fraud analyst workflows.

## Future Improvements

- Add FastAPI scoring API
- Add XGBoost or LightGBM
- Add SHAP explainability
- Add real-time streaming simulation
- Deploy using Streamlit Cloud
- Add model monitoring and drift detection

## Author

Sayli Anandrao Patil
