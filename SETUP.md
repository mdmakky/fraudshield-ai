# FraudShield AI - Setup Guide

## Step-by-Step Installation

### 1. Download the Dataset

Before running any code, you need the Kaggle Credit Card Fraud Detection dataset:

1. Go to: https://www.kaggle.com/mlg-ulb/creditcardfraud
2. Click "Download" (requires Kaggle account)
3. Extract `creditcard.csv`
4. Place it in: `fraudshield-ai/data/creditcard.csv`

### 2. Install Dependencies

```bash
# Make sure you're in the project directory
cd fraudshield-ai

# Install all required packages
pip install -r requirements.txt
```

This will install:
- Data science: pandas, numpy, scikit-learn
- ML: xgboost, imbalanced-learn
- Deep learning: tensorflow
- Explainability: shap
- API: fastapi, uvicorn
- Visualization: matplotlib, seaborn

### 3. Train Models

#### Option A: Using Jupyter Notebooks (Recommended)

```bash
jupyter notebook
```

Then open and run:
- `notebooks/01-fraud-detection-complete.ipynb` - Full supervised learning pipeline
- `notebooks/02-autoencoder-anomaly.ipynb` - Unsupervised anomaly detection

#### Option B: Using Python Scripts

```bash
# Train supervised models (LR, RF, XGBoost)
python src/train_models.py

# Train autoencoder
python src/autoencoder.py

# Generate SHAP explanations
python src/explain.py
```

After training, you'll have models in the `models/` directory:
- `lr.joblib` - Logistic Regression
- `rf.joblib` - Random Forest
- `xgb.joblib` - XGBoost
- `autoencoder.h5` - Autoencoder
- `scaler.joblib` - Feature scaler
- `model_comparison.csv` - Performance comparison

### 4. Run the API

```bash
# Start the FastAPI server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Visit:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 5. Test the API

```bash
# In a new terminal
python test_api.py
```

Or use curl:
```bash
curl http://localhost:8000/health
```

---

## Docker Deployment

### Option 1: Docker Build

```bash
# Build image
docker build -t fraudshield-ai .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  fraudshield-ai
```

### Option 2: Docker Compose

```bash
# Start service
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop service
docker-compose down
```

---

## Troubleshooting

### Issue: "Module not found"
```bash
# Make sure you activated the virtual environment
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Then reinstall
pip install -r requirements.txt
```

### Issue: "Data file not found"
```bash
# Make sure creditcard.csv is in the right place:
# fraudshield-ai/data/creditcard.csv

# Check if file exists (Windows):
dir data\creditcard.csv

# Check if file exists (Linux/Mac):
ls -lh data/creditcard.csv
```

### Issue: API won't start
```bash
# Check if port 8000 is already in use
# Windows:
netstat -ano | findstr :8000

# Linux/Mac:
lsof -i :8000

# Use a different port:
uvicorn src.api:app --port 8080
```

### Issue: Out of memory during training
```python
# Edit src/train_models.py, reduce sample size:
# Change:
X_sample = X_test.sample(n=1000, random_state=42)
# To:
X_sample = X_test.sample(n=500, random_state=42)
```

---

## Next Steps

1. ✅ Run the notebooks to understand the methodology
2. ✅ Train models with your own hyperparameters
3. ✅ Test the API with sample transactions
4. ✅ Deploy to cloud (AWS, GCP, Azure)
5. ✅ Set up monitoring and alerting
6. ✅ Implement continuous retraining

---

## Performance Tips

### Training Speed
- Use `n_jobs=-1` for parallel processing
- Use GPU for TensorFlow (autoencoder)
- Reduce SMOTE sample size if too slow

### Inference Speed
- Keep models loaded in memory (done in API)
- Use batch predictions for multiple transactions
- Consider ONNX runtime for faster inference

### Memory Usage
- Use `del` to free memory after training
- Process data in chunks for large datasets
- Use `gc.collect()` to force garbage collection

---

## Common Workflows

### Daily Development
```bash
# Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run notebooks for exploration
jupyter notebook

# Test changes
python src/train_models.py

# Start API for testing
uvicorn src.api:app --reload
```

### Model Retraining
```bash
# Update data
cp /path/to/new/creditcard.csv data/

# Retrain
python src/train_models.py
python src/autoencoder.py

# Evaluate
python src/explain.py

# Restart API (automatically picks up new models)
```

### Production Deployment
```bash
# Build production image
docker build -t fraudshield-ai:v1.0 .

# Tag for registry
docker tag fraudshield-ai:v1.0 your-registry/fraudshield-ai:v1.0

# Push to registry
docker push your-registry/fraudshield-ai:v1.0

# Deploy to Kubernetes/ECS/etc
kubectl apply -f k8s/deployment.yaml
```

---

**Ready to detect fraud! 🛡️**
