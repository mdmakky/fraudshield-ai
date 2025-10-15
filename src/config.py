"""
Configuration settings for FraudShield AI.
"""

# Data settings
DATA_PATH = 'data/creditcard.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature engineering
FEATURES_TO_SCALE = ['Amount', 'Amount_log', 'Hour', 'Day', 'Amount_bin']

# SMOTE settings
SMOTE_RANDOM_STATE = 42
SMOTE_K_NEIGHBORS = 5

# Model hyperparameters
LOGISTIC_REGRESSION = {
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

RANDOM_FOREST = {
    'n_estimators': 200,
    'class_weight': 'balanced',
    'max_depth': 10,
    'min_samples_split': 10,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

XGBOOST = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'auc',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Autoencoder settings
AUTOENCODER = {
    'latent_dim': 8,
    'hidden_layers': [64, 32],
    'epochs': 50,
    'batch_size': 256,
    'validation_split': 0.1,
    'early_stopping_patience': 5
}

# Evaluation settings
TARGET_RECALL = 0.90
DEFAULT_THRESHOLD = 0.5

# SHAP settings
SHAP_SAMPLE_SIZE = 1000
SHAP_MAX_DISPLAY = 20

# API settings
API_HOST = '0.0.0.0'
API_PORT = 8000
API_RELOAD = True  # Set to False in production

# Model paths
MODEL_DIR = 'models'
MODEL_PATHS = {
    'lr': f'{MODEL_DIR}/lr.joblib',
    'rf': f'{MODEL_DIR}/rf.joblib',
    'xgb': f'{MODEL_DIR}/xgb.joblib',
    'autoencoder': f'{MODEL_DIR}/autoencoder.h5',
    'scaler': f'{MODEL_DIR}/scaler.joblib',
    'ae_threshold': f'{MODEL_DIR}/ae_threshold.joblib',
    'comparison': f'{MODEL_DIR}/model_comparison.csv'
}

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
