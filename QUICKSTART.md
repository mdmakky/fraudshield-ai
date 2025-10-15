# 🚀 Quick Start Guide

Get FraudShield AI running in 5 minutes!

## Prerequisites
- Python 3.10+
- 500 MB free disk space
- Internet connection (for downloading dataset)

## Installation

### 1. Download Dataset (Required)

Visit https://www.kaggle.com/mlg-ulb/creditcardfraud and download `creditcard.csv`

Place it in: `fraudshield-ai/data/creditcard.csv`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Choose Your Path

### 🎓 Path A: Learn with Notebooks (Recommended for First-Time)

```bash
jupyter notebook
```

Open and run in order:
1. `notebooks/01-fraud-detection-complete.ipynb` ⭐ Start here!
2. `notebooks/02-autoencoder-anomaly.ipynb`

**Time**: ~30 minutes (includes training)

**What you'll learn**:
- How fraud detection models work
- Feature engineering techniques
- SMOTE for imbalanced data
- Threshold tuning for high recall
- SHAP explainability
- Autoencoder anomaly detection

---

### ⚡ Path B: Quick Train & Deploy

```bash
# Train models (5-10 minutes)
python src/train_models.py

# Start API
uvicorn src.api:app --reload
```

Visit: http://localhost:8000/docs

**Time**: ~10 minutes

---

### 🐳 Path C: Docker Deployment

```bash
# Build and run
docker-compose up -d

# Test
curl http://localhost:8000/health
```

**Time**: ~5 minutes (after models are trained)

---

## Test Your Setup

```bash
python test_api.py
```

Expected output:
```
FraudShield AI - API Test
==============================================================

Testing health endpoint...
Status: 200
Response: {
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "threshold": 0.3
}

Testing legitimate transaction...
Status: 200
Response:
  Fraud Score: 0.0234
  Is Fraud: False
  Risk Level: low
  Message: ✓ Transaction appears legitimate (score: 2.34%)

...

All tests completed successfully! ✓
```

---

## What's Next?

### Explore the API
- **Swagger UI**: http://localhost:8000/docs
- **Try predictions**: Click "Try it out" on `/predict` endpoint
- **Batch processing**: Test `/batch-predict` endpoint

### Understand the Models
```bash
# View model comparison
cat models/model_comparison.csv

# Generate SHAP explanations
python src/explain.py
```

### Customize
Edit `src/config.py` to adjust:
- Hyperparameters
- Threshold values
- Feature engineering
- SMOTE settings

### Deploy to Production
See `README.md` section "🐳 Docker Deployment" for:
- Kubernetes deployment
- Cloud hosting (AWS, GCP, Azure)
- Monitoring and logging
- API security

---

## Common Commands

```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Start Jupyter
jupyter notebook

# Train models
python src/train_models.py
python src/autoencoder.py

# Start API
uvicorn src.api:app --reload

# Run tests
python test_api.py

# Docker
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## Need Help?

1. **Check Setup Guide**: `SETUP.md`
2. **Full Documentation**: `README.md`
3. **Troubleshooting**: `SETUP.md` → Troubleshooting section
4. **Open an issue**: GitHub Issues

---

## Project Structure

```
fraudshield-ai/
├── data/
│   └── creditcard.csv          ← Place dataset here
├── notebooks/
│   ├── 01-fraud-detection-complete.ipynb  ← Start here!
│   └── 02-autoencoder-anomaly.ipynb
├── src/
│   ├── train_models.py         ← Train supervised models
│   ├── autoencoder.py          ← Train autoencoder
│   ├── api.py                  ← FastAPI service
│   └── explain.py              ← SHAP explanations
├── models/                     ← Generated after training
├── requirements.txt
└── README.md
```

---

**Ready to detect fraud!** 🛡️

Start with the notebooks, then move to API deployment.

Questions? Check `README.md` or open an issue on GitHub.
