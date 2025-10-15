# Models Directory

This directory stores trained models and artifacts.

## Files Generated After Training

After running `src/train_models.py`:
- `lr.joblib` - Logistic Regression model
- `rf.joblib` - Random Forest model
- `xgb.joblib` - XGBoost model (typically best performance)
- `scaler.joblib` - RobustScaler for feature normalization
- `model_comparison.csv` - Performance metrics comparison
- `pr_curve.png` - Precision-Recall curves visualization

After running `src/autoencoder.py`:
- `autoencoder.h5` - Keras autoencoder model
- `ae_threshold.joblib` - Optimal threshold for anomaly detection
- `ae_error_dist.png` - Reconstruction error distribution
- `ae_training_history.png` - Training history plots

After running `src/explain.py`:
- `shap_summary_*.png` - SHAP summary plots for each model
- `shap_force_*.png` - Force plots for individual predictions
- `shap_dependence_*.png` - Dependence plots for top features

## Model Loading

Models are loaded automatically by the FastAPI service (`src/api.py`).

To load models manually:

```python
import joblib
from tensorflow import keras

# Load supervised models
lr = joblib.load('models/lr.joblib')
rf = joblib.load('models/rf.joblib')
xgb = joblib.load('models/xgb.joblib')
scaler = joblib.load('models/scaler.joblib')

# Load autoencoder
autoencoder = keras.models.load_model('models/autoencoder.h5')
ae_threshold = joblib.load('models/ae_threshold.joblib')
```

## Model Versioning

For production, consider:
- Timestamped model files: `xgb_2024_01_15.joblib`
- Model registry (MLflow, Weights & Biases)
- Git LFS for model version control
- Separate model storage (S3, GCS, Azure Blob)

---

**Models will appear here after training** 🎯
