"""
FraudShield AI - Credit Card Fraud Detection System

A complete machine learning system for detecting credit card fraud using
supervised learning (Logistic Regression, Random Forest, XGBoost) and
unsupervised learning (Autoencoder-based anomaly detection).

Modules:
    data_prep: Data loading and feature engineering
    train_models: Supervised model training with SMOTE
    autoencoder: Unsupervised anomaly detection
    explain: SHAP-based model explainability
    api: FastAPI service for real-time predictions
    config: Configuration settings
"""

__version__ = '1.0.0'
__author__ = 'Makky MD'
__email__ = 'your-email@example.com'

from . import data_prep
from . import config

__all__ = ['data_prep', 'config']
