# FraudShield AI 🛡️

> **Enterprise-grade credit card fraud detection system** using machine learning and deep learning techniques.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
- **Deep Learning**: Autoencoder-based anomaly detection (unsupervised)
- **Class Imbalance Handling**: SMOTE oversampling
- **Explainability**: SHAP values for model interpretability
- **Real-time API**: FastAPI service for production deployment
- **Threshold Tuning**: Optimized for high recall (90%+)
- **Comprehensive Notebooks**: Step-by-step analysis and training

## 📊 Results

| Model | ROC-AUC | PR-AUC | Optimal Recall | Optimal Precision |
|-------|---------|--------|----------------|-------------------|
| **XGBoost** | 0.98+ | 0.85+ | 0.90+ | 0.80+ |
| Random Forest | 0.97+ | 0.82+ | 0.90+ | 0.75+ |
| Logistic Regression | 0.95+ | 0.75+ | 0.90+ | 0.65+ |
| Autoencoder | 0.94+ | 0.70+ | 0.90+ | 0.60+ |

*Results may vary depending on dataset and hyperparameters*

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker for containerized deployment

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/mdmakky/fraudshield-ai.git
cd fraudshield-ai
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**

Download the Kaggle Credit Card Fraud Detection dataset:
- URL: https://www.kaggle.com/mlg-ulb/creditcardfraud
- Place `creditcard.csv` in the `data/` directory

---

## 📖 Usage

### Option 1: Jupyter Notebooks (Recommended for Learning)

#### Complete Fraud Detection Pipeline
```bash
jupyter notebook notebooks/01-fraud-detection-complete.ipynb
```

This notebook covers:
- ✅ Data exploration and visualization
- ✅ Feature engineering
- ✅ SMOTE for class imbalance
- ✅ Training 3 supervised models
- ✅ Threshold tuning for high recall
- ✅ SHAP explainability
- ✅ Model comparison and evaluation

#### Autoencoder Anomaly Detection
```bash
jupyter notebook notebooks/02-autoencoder-anomaly.ipynb
```

This notebook demonstrates:
- ✅ Unsupervised learning approach
- ✅ Training autoencoder on legitimate transactions only
- ✅ Anomaly detection via reconstruction error
- ✅ Threshold optimization
- ✅ Comparison with supervised methods

### Option 2: Python Scripts

#### Train All Models
```bash
python src/train_models.py
```

This will:
1. Load and prepare data
2. Apply SMOTE
3. Train Logistic Regression, Random Forest, XGBoost
4. Evaluate and compare models
5. Save models to `models/` directory
6. Generate performance plots

#### Train Autoencoder
```bash
python src/autoencoder.py
```

#### Generate SHAP Explanations
```bash
python src/explain.py
```

### Option 3: FastAPI Service (Production)

#### Start API Server
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

#### API Documentation
Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### Example API Request

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0,
    "Amount": 150.00,
    "V1": -1.359807,
    "V2": -0.072781,
    ...
    "V28": -0.021053
  }'
```

**Response:**
```json
{
  "fraud_score": 0.0234,
  "is_fraud": false,
  "risk_level": "low",
  "threshold": 0.3,
  "message": "✓ Transaction appears legitimate (score: 2.34%)"
}
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t fraudshield-ai .

# Run container
docker run -p 8000:8000 fraudshield-ai
```

### Using Docker Compose

```bash
docker-compose up -d
```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## 📁 Project Structure

```
fraudshield-ai/
├── data/
│   └── creditcard.csv          # Kaggle dataset (download separately)
├── notebooks/
│   ├── 01-fraud-detection-complete.ipynb
│   └── 02-autoencoder-anomaly.ipynb
├── src/
│   ├── data_prep.py            # Data loading & feature engineering
│   ├── train_models.py         # Supervised model training
│   ├── autoencoder.py          # Autoencoder anomaly detection
│   ├── explain.py              # SHAP explainability
│   └── api.py                  # FastAPI service
├── models/                     # Saved models (generated after training)
│   ├── lr.joblib
│   ├── rf.joblib
│   ├── xgb.joblib
│   ├── autoencoder.h5
│   ├── scaler.joblib
│   └── model_comparison.csv
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔬 Methodology

### 1. Data Preparation
- **Feature Engineering**: Log-transform amount, extract time features (hour, day)
- **Scaling**: RobustScaler for outlier-resistant normalization
- **Train-Test Split**: 80/20 stratified split

### 2. Class Imbalance Handling
- **SMOTE**: Synthetic Minority Oversampling Technique
- **Class Weights**: Applied in Logistic Regression and Random Forest
- **Scale Pos Weight**: XGBoost parameter for imbalanced data

### 3. Model Training

#### Supervised Models:
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting (typically best performance)

#### Unsupervised Model:
- **Autoencoder**: Neural network trained on legitimate transactions only
- Detects fraud as anomalies based on reconstruction error

### 4. Evaluation Metrics

**Primary Metrics** (for imbalanced data):
- **PR-AUC** (Precision-Recall AUC): More informative than ROC-AUC
- **Recall**: Priority for fraud detection (minimize false negatives)
- **Precision**: Balance to reduce false alarms

**Secondary Metrics**:
- ROC-AUC
- F1-Score
- Confusion Matrix

### 5. Threshold Tuning
- Optimize threshold for **90%+ recall**
- Accept lower precision to catch more fraud
- Business decision: cost of false negative >> cost of false positive

### 6. Explainability
- **SHAP** (SHapley Additive exPlanations)
- Feature importance for model decisions
- Individual prediction explanations
- Helps reduce false positives by understanding model reasoning

---

## 🎓 Key Insights

### Feature Importance (from SHAP analysis)
Top features for fraud detection:
1. **V14, V12, V10**: PCA features (specific patterns)
2. **Amount_log**: Transaction amount (log-scaled)
3. **V17, V4**: Additional PCA features
4. **Hour**: Time of day patterns

### Fraud Patterns Detected
- ✅ Unusual transaction amounts
- ✅ Abnormal time patterns
- ✅ Specific V-feature combinations (PCA-transformed)
- ✅ Deviation from normal spending behavior

### Model Selection Guidance

**Use XGBoost when:**
- You have labeled training data
- Need highest accuracy
- Can retrain periodically

**Use Autoencoder when:**
- Limited fraud examples
- Fraud patterns evolve rapidly
- Want to detect novel fraud types
- Prefer unsupervised approach

**Best Strategy: Ensemble**
- Run both XGBoost and Autoencoder
- Flag transaction if EITHER model predicts fraud
- Maximizes recall, catches more fraud types

---

## 📊 Monitoring & Production

### Model Monitoring
Track these metrics in production:

1. **Performance Metrics**
   - Daily/weekly recall, precision, F1
   - False positive rate
   - Alert-to-fraud conversion rate

2. **Data Drift**
   - Feature distributions over time
   - Amount statistics
   - Time patterns

3. **Business Metrics**
   - Fraud caught (true positives)
   - Fraud missed (false negatives)
   - Manual review load (false positives)
   - Cost savings

### Retraining Strategy

**Trigger retrain when:**
- PR-AUC drops below threshold (e.g., 0.75)
- Significant data drift detected
- Monthly scheduled retraining
- New fraud patterns emerge

**Retraining Process:**
1. Collect recent labeled data
2. Re-run feature engineering
3. Retrain models with updated data
4. A/B test new model vs current
5. Deploy if performance improves

---

## 🔐 Security & Compliance

### Data Privacy
- ✅ No PII stored (dataset already anonymized)
- ✅ Use tokenization for card numbers
- ✅ Encrypt data at rest and in transit
- ✅ Apply access controls (RBAC)

### Compliance
- **PCI-DSS**: For card data handling
- **GDPR**: If processing EU customer data
- **SOC 2**: For service organization controls

### API Security
- 🔒 Implement JWT authentication
- 🔒 Rate limiting to prevent abuse
- 🔒 Input validation
- 🔒 HTTPS only in production
- 🔒 API key rotation

---

## 🛠️ Advanced Features

### Threshold Configuration
Update threshold dynamically via API:

```bash
curl -X POST "http://localhost:8000/update-threshold" \
  -H "Content-Type: application/json" \
  -d '{"new_threshold": 0.25}'
```

### Batch Predictions
Process multiple transactions:

```python
import requests

transactions = {
    "transactions": [
        {...},  # Transaction 1
        {...},  # Transaction 2
        # ... more transactions
    ]
}

response = requests.post("http://localhost:8000/batch-predict", json=transactions)
results = response.json()
print(f"Fraud count: {results['fraud_count']}/{results['total_processed']}")
```

---

## 📈 Performance Optimization

### Training Speed
- Use `n_jobs=-1` for parallel processing
- Reduce SMOTE sample size for faster training
- Use GPU for TensorFlow/Keras (autoencoder)

### Inference Speed
- Keep models loaded in memory (API)
- Batch predictions when possible
- Consider ONNX for production inference
- Use model quantization for edge deployment

### Scalability
- Deploy multiple API instances (load balancing)
- Use Redis for caching
- Asynchronous prediction queue (Celery + Redis)
- Database for logging predictions

---

## 🧪 Testing

### Run Tests (TODO)
```bash
pytest tests/
```

### Test API Locally
```bash
# Install testing tools
pip install httpx pytest

# Run test script
python tests/test_api.py
```

---

## 📚 Additional Resources

### Dataset
- [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions (492 frauds, 0.172%)
- PCA-transformed features for privacy

### Research Papers
1. [Credit Card Fraud Detection: A Realistic Modeling](https://www.researchgate.net/publication/260837261)
2. [Deep Learning for Fraud Detection](https://arxiv.org/abs/1906.07930)
3. [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)

### Tools & Libraries
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [imbalanced-learn](https://imbalanced-learn.org/)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Makky MD**
- GitHub: [@mdmakky](https://github.com/mdmakky)

---

## 🙏 Acknowledgments

- Kaggle for the Credit Card Fraud Detection dataset
- The open-source ML/DL community
- FastAPI, XGBoost, SHAP, and TensorFlow teams

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## 🗺️ Roadmap

- [ ] Add unit tests and integration tests
- [ ] Implement A/B testing framework
- [ ] Add Prometheus metrics for monitoring
- [ ] Create Grafana dashboards
- [ ] Support for additional fraud types (account takeover, etc.)
- [ ] Real-time streaming with Kafka
- [ ] Model versioning with MLflow
- [ ] Feature store integration
- [ ] Explainability dashboard
- [ ] Mobile app for fraud analysts

---

**Built with ❤️ for safer financial transactions**
