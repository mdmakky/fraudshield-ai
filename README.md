# FraudShield AI 🛡️# FraudShield AI 🛡️# FraudShield AI 🛡️



**AI-Powered Fraud Detection for Financial Transactions**



Real-time fraud detection API using XGBoost with 97%+ accuracy on 5 million transactions.**AI-Powered Fraud Detection for Financial Transactions****AI-Powered Fraud Detection for Financial Transactions**



---



## ⚡ Quick Run (If You Have Models)Real-time fraud detection API using XGBoost with 97%+ accuracy on 5 million transactions.



**Already have trained models in the `models/` folder?** Just run:



```bash------

# 1. Install dependencies

pip install -r requirements.txt



# 2. Start the API## 🚀 Quick Start

python api.py

# or

uvicorn api:app --host 0.0.0.0 --port 8000

```**1. Setup**## 📊 Overview## FeaturesDetect fraudulent transactions using Machine Learning with Logistic Regression, Random Forest, and XGBoost models.



That's it! Your API is ready at `http://localhost:8000````bash



Visit `http://localhost:8000/docs` for interactive documentation.# Install dependencies



---pip install -r requirements.txt



## 🚀 Full Setup (From Scratch)```- **Dataset**: 5,000,000 financial transactions from Kaggle



**1. Install Dependencies**

```bash

pip install -r requirements.txt**2. Get Dataset**- **Fraud Rate**: 3.59% (highly imbalanced)

```

- Download from [Kaggle](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection)

**2. Get Dataset**

- Download from [Kaggle](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection)- Place `financial_fraud_detection_dataset.csv` in `data/` folder- **Model**: XGBoost Classifier with 300 estimators- **97%+ Accuracy**: Ensemble of XGBoost, LightGBM, and CatBoost---

- Place `financial_fraud_detection_dataset.csv` in `data/` folder



**3. Train Model**

```bash**3. Train Model**- **Performance**: 99.11% ROC-AUC Score

python train.py

``````bash

This will create trained models in the `models/` folder (~5-10 minutes).

python train.py- **Features**: 50 engineered features from 26 base features- **Advanced Feature Engineering**: Statistical aggregations, interaction features, time-based features

**4. Start API**

```bash```

python api.py

# or- **API**: FastAPI REST service for real-time predictions

uvicorn api:app --host 0.0.0.0 --port 8000

```**4. Start API**



**5. Test It**```bash- **Memory Optimized**: Handles large datasets efficiently on 16GB RAM## 📊 Quick Overview

```bash

curl http://localhost:8000/healthpython api.py

```

# or---

Visit `http://localhost:8000/docs` for interactive API documentation.

uvicorn api:app --host 0.0.0.0 --port 8000

---

```- **REST API**: FastAPI-based prediction service

## 📊 What It Does



- **Detects** fraudulent transactions in real-time

- **Achieves** 97%+ ROC-AUC score**5. Test It**## 🚀 Quick Start

- **Analyzes** 50+ features per transaction

- **Provides** fraud probability and confidence level```bash



---curl http://localhost:8000/health- **Production Ready**: Optimized hyperparameters and preprocessing- **Dataset**: 5M financial transactions from Kaggle



## 🔧 API Usage```



**Check Health:**### Prerequisites

```bash

curl http://localhost:8000/healthVisit `http://localhost:8000/docs` for interactive API documentation.

```

- **Models**: 3 ML models (LR, RF, XGBoost)

**Predict Fraud:**

```bash---

curl -X POST "http://localhost:8000/predict" \

  -H "Content-Type: application/json" \- Python 3.8+

  -d '{

    "timestamp": "2024-01-15T10:30:00",## 📊 What It Does

    "sender_account": "ACC123",

    "receiver_account": "ACC456",- 16GB RAM (minimum)## Quick Start- **Features**: 18 columns (transaction details, behavioral scores, metadata)

    "amount": 1500.0,

    "transaction_type": "transfer",- **Detects** fraudulent transactions in real-time

    "merchant_category": "online",

    "location": "New York",- **Achieves** 97%+ ROC-AUC score- Ubuntu/Linux (recommended) or Windows/macOS

    "device_used": "mobile",

    "payment_channel": "app",- **Analyzes** 50+ features per transaction

    "time_since_last_transaction": 300.0,

    "spending_deviation_score": 3.5,- **Provides** fraud probability and confidence level- **API**: FastAPI REST service for real-time predictions

    "velocity_score": 18.0,

    "geo_anomaly_score": 0.9

  }'

```---### 1. Clone the Repository



**Response:**

```json

{## 🔧 API Usage1. **Install dependencies:**- **Goal**: Detect fraud with 90%+ recall

  "is_fraud": true,

  "fraud_probability": 0.94,

  "confidence": "high"

}**Check Health:**```bash

```

```bash

---

curl http://localhost:8000/healthgit clone <your-repo-url>   ```bash

## 📁 Project Structure

```

```

fraudshield-ai/cd fraudshield-ai

├── api.py                 # FastAPI server

├── train.py              # Model training**Predict Fraud:**

├── requirements.txt      # Dependencies

├── data/                 # Dataset folder```bash```   pip install -r requirements.txt---

│   └── financial_fraud_detection_dataset.csv

└── models/               # Trained models (from GitHub or training)curl -X POST "http://localhost:8000/predict" \

    ├── fraud_detection_model.joblib

    ├── label_encoders.joblib  -H "Content-Type: application/json" \

    └── feature_columns.joblib

```  -d '{



---    "timestamp": "2024-01-15T10:30:00",### 2. Create Virtual Environment   ```



## 🎯 Model Performance    "sender_account": "ACC123",



- **ROC-AUC:** 97%+    "receiver_account": "ACC456",

- **Precision:** 95%+  

- **Recall:** 90%+    "amount": 1500.0,

- **Algorithm:** XGBoost with 300 estimators

- **Features:** 50+ engineered features    "transaction_type": "transfer",```bash## 🚀 Quick Start (5 Minutes)

- **Dataset:** 5M transactions

    "merchant_category": "online",

---

    "location": "New York",# Create virtual environment

## 🔍 Key Features

    "device_used": "mobile",

- Advanced feature engineering (time-based, statistical aggregations)

- Balanced dataset (2:1 ratio with SMOTE)    "payment_channel": "app",python3 -m venv fraudshield-env2. **Place your data:**

- Network analysis (account relationships)

- Risk scoring (velocity, deviation, geo-anomaly)    "time_since_last_transaction": 300.0,

- Production-ready REST API

    "spending_deviation_score": 3.5,

---

    "velocity_score": 18.0,

## 🛠️ Tech Stack

    "geo_anomaly_score": 0.9# Activate virtual environment   - Put `financial_fraud_detection_dataset.csv` in the `data/` directory### 1. Setup Environment

- **Python 3.8+**

- **XGBoost** - Gradient boosting  }'

- **FastAPI** - REST API

- **Pandas/NumPy** - Data processing```# On Linux/macOS:

- **Scikit-learn** - ML utilities



---

**Response:**source fraudshield-env/bin/activate

## 📖 Requirements

```json

```

pandas>=2.0.0{

numpy>=1.24.0

scikit-learn>=1.3.0  "is_fraud": true,

xgboost>=2.0.0

fastapi>=0.100.0  "fraud_probability": 0.94,# On Windows:3. **Train the model:**```bash

uvicorn>=0.20.0

```  "confidence": "high"



---}fraudshield-env\Scripts\activate



## 💡 Common Issues```



**Port already in use:**```   ```bash# Install dependencies

```bash

kill $(lsof -t -i:8000)---

```



**Model not found:**

```bash## 📁 Project Structure

python train.py  # Train models first

```### 3. Install Dependencies   python train.pypip install -r requirements.txt



**Import errors:**```

```bash

pip install -r requirements.txtfraudshield-ai/

```

├── api.py                 # FastAPI server

---

├── train.py              # Model training```bash   ``````

## 📝 License

├── requirements.txt      # Dependencies

MIT License

├── data/                 # Dataset folderpip install -r requirements.txt

---

│   └── financial_fraud_detection_dataset.csv

**Built with ❤️ for secure financial transactions**

└── models/               # Trained models (auto-generated)```

    ├── fraud_detection_model.joblib

    ├── label_encoders.joblib

    └── feature_columns.joblib

```### 4. Download Dataset4. **Start the API:**### 2. Download Dataset



---



## 🎯 Model Performance1. Download the dataset from [Kaggle - Financial Fraud Detection Dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)   ```bash



- **ROC-AUC:** 97%+2. Place `financial_fraud_detection_dataset.csv` in the `data/` directory

- **Precision:** 95%+  

- **Recall:** 90%+   python api.py1. Go to: https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection

- **Algorithm:** XGBoost with 300 estimators

- **Features:** 50+ engineered features```bash

- **Dataset:** 5M transactions

mkdir -p data   ```2. Download `financial_fraud_detection_dataset.csv` (~796 MB)

---

# Place your CSV file in data/

## 🔍 Key Features

```3. Place in `data/` folder:

- Advanced feature engineering (time-based, statistical aggregations)

- Balanced dataset (2:1 ratio with SMOTE)

- Network analysis (account relationships)

- Risk scoring (velocity, deviation, geo-anomaly)### 5. Train the Model5. **Make predictions:**   ```bash

- Production-ready REST API



---

```bash   ```bash   mv ~/Downloads/financial_fraud_detection_dataset.csv data/

## 🛠️ Tech Stack

python train.py

- **Python 3.8+**

- **XGBoost** - Gradient boosting```   curl -X POST "http://localhost:8000/predict" \   ```

- **FastAPI** - REST API

- **Pandas/NumPy** - Data processing

- **Scikit-learn** - ML utilities

**Training Process:**     -H "Content-Type: application/json" \

---

- Loads 5M transactions

## 📖 Requirements

- Creates 50 advanced features     -d '{### 3. Train Models

```

pandas>=2.0.0- Balances dataset (2:1 ratio)

numpy>=1.24.0

scikit-learn>=1.3.0- Trains XGBoost model (~5 minutes)       "timestamp": "2024-01-01 12:00:00",

xgboost>=2.0.0

fastapi>=0.100.0- Saves model to `models/` directory

uvicorn>=0.20.0

```       "sender_account": "acc123",```bash



---**Expected Output:**



## 💡 Common Issues```       "receiver_account": "acc456",# Quick test (100k samples, ~5 minutes)



**Port already in use:**=== FraudShield AI - High Accuracy Training ===

```bash

kill $(lsof -t -i:8000)...       "amount": 1000.0,python src/train_models.py --sample 100000

```

ROC-AUC Score: 0.9911

**Model not found:**

```bash🎉 ACHIEVED TARGET ACCURACY! 🎉       "transaction_type": "transfer",

python train.py  # Train first

``````



---       "merchant_category": "online_shopping",# Full training (~30-60 minutes)



## 📝 License### 6. Start the API Server



MIT License       "location": "New York",python src/train_models.py



---```bash



**Built with ❤️ for secure financial transactions**python3 -m uvicorn api:app --host 0.0.0.0 --port 8000       "device_used": "mobile",```


```

       "payment_channel": "app",

Or run in background:

```bash       "time_since_last_transaction": 3600.0,This creates:

nohup python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

```       "spending_deviation_score": 2.5,- `models/lr.joblib` - Logistic Regression



The API will be available at: `http://localhost:8000`       "velocity_score": 15.0,- `models/rf.joblib` - Random Forest



---       "geo_anomaly_score": 0.8- `models/xgb.joblib` - XGBoost



## 📡 API Usage     }'- `models/scaler.joblib` - Data scaler



### Health Check   ```- `models/label_encoders.joblib` - Encoders



```bash- `models/model_comparison.csv` - Performance metrics

curl http://localhost:8000/health

```## Project Structure



**Response:**### 4. Start API Server

```json

{```

  "status": "healthy",

  "model_loaded": true,fraudshield-ai/```bash

  "timestamp": "2025-11-17T00:00:00"

}├── train.py              # Main training scriptpython src/api.py

```

├── api.py                # FastAPI prediction service```

### Predict Fraud

├── requirements.txt      # Python dependencies

```bash

curl -X POST "http://localhost:8000/predict" \├── data/                 # Dataset directoryVisit: http://localhost:8000/docs

  -H "Content-Type: application/json" \

  -d '{└── models/              # Trained models and preprocessing objects

    "timestamp": "2023-12-01T02:15:30.000000",

    "sender_account": "ACC999999",```### 5. Test Prediction

    "receiver_account": "ACC111111",

    "amount": 2500.00,

    "transaction_type": "transfer",

    "merchant_category": "online",## Model Performance```bash

    "location": "Dubai",

    "device_used": "mobile",python test_api.py

    "payment_channel": "wire_transfer",

    "time_since_last_transaction": 0.0,- **ROC-AUC**: 97%+```

    "spending_deviation_score": 4.50,

    "velocity_score": 18,- **Precision**: 95%+

    "geo_anomaly_score": 0.95

  }'- **Recall**: 90%+Or use curl:

```

- **F1-Score**: 92%+```bash

**Response:**

```jsoncurl -X POST "http://localhost:8000/predict?model=xgboost" \

{

  "is_fraud": true,## Key Techniques  -H "Content-Type: application/json" \

  "fraud_probability": 0.9739,

  "confidence": "high"  -d '{

}

```1. **Ensemble Learning**: XGBoost + LightGBM + CatBoost with soft voting    "transaction_id": "TEST001",



### Interactive API Documentation2. **Advanced Features**: Statistical aggregations by merchant, device, and location    "timestamp": "2024-01-15T10:30:00",



Visit `http://localhost:8000/docs` in your browser for Swagger UI with interactive testing.3. **Class Balancing**: Built-in class weights and optional SMOTE    "sender_account": "ACC123",



---4. **Hyperparameter Optimization**: Carefully tuned for fraud detection    "receiver_account": "ACC456",



## 🧪 Test Cases5. **Memory Optimization**: Efficient preprocessing for large datasets    "amount": 500.0,

    "transaction_type": "transfer",

### Legitimate Transaction    "time_since_last_transaction": 3600,

```json    "spending_deviation_score": 1.5,

{    "velocity_score": 0.4,

  "timestamp": "2023-08-22T09:22:43.516168",    "geo_anomaly_score": 0.2,

  "sender_account": "ACC877572",    "location": "New York",

  "receiver_account": "ACC388389",    "device": "mobile",

  "amount": 25.50,    "payment_channel": "app"

  "transaction_type": "withdrawal",  }'

  "merchant_category": "utilities",```

  "location": "Tokyo",

  "device_used": "mobile",---

  "payment_channel": "card",

  "time_since_last_transaction": 3600.0,## 📁 Project Structure

  "spending_deviation_score": -0.21,

  "velocity_score": 3,```

  "geo_anomaly_score": 0.22fraudshield-ai/

}├── data/

```│   └── financial_fraud_detection_dataset.csv  # Download from Kaggle

**Expected Result:** `is_fraud: false`, probability < 0.001├── models/                                     # Trained models (created after training)

├── src/

### Fraudulent Transaction│   ├── config.py          # Configuration settings

```json│   ├── data_prep.py       # Data preprocessing

{│   ├── train_models.py    # Train models

  "timestamp": "2023-12-15T03:10:45.000000",│   ├── autoencoder.py     # Autoencoder (optional)

  "sender_account": "ACC000000",│   └── api.py             # API server

  "receiver_account": "ACC999999",├── requirements.txt       # Python dependencies

  "amount": 4500.00,├── test_api.py           # Test script

  "transaction_type": "transfer",└── README.md             # This file

  "merchant_category": "online",```

  "location": "Singapore",

  "device_used": "mobile",---

  "payment_channel": "UPI",

  "time_since_last_transaction": 15.0,## 📊 Dataset Features

  "spending_deviation_score": 4.80,

  "velocity_score": 19,**18 columns total:**

  "geo_anomaly_score": 0.98

}**Transaction Details** (6):

```- transaction_id, timestamp, sender_account, receiver_account, amount, transaction_type

**Expected Result:** `is_fraud: true`, probability > 0.65

**Behavioral Features** (4):

---- time_since_last_transaction, spending_deviation_score, velocity_score, geo_anomaly_score



## 🏗️ Project Structure**Metadata** (5):

- location, device, payment_channel, ip_address, device_hash

```

fraudshield-ai/**Target** (2):

├── api.py                      # FastAPI REST API- is_fraud (0/1), fraud_type (type of fraud)

├── train.py                    # Model training script

├── requirements.txt            # Python dependencies---

├── README.md                   # This file

├── data/## 🎯 Model Performance

│   └── financial_fraud_detection_dataset.csv

├── models/Expected results (may vary):

│   ├── fraud_detection_model.joblib

│   ├── label_encoders.joblib| Model | ROC-AUC | PR-AUC | Recall | Precision |

│   └── feature_columns.joblib|-------|---------|--------|--------|-----------|

└── fraudshield-env/            # Virtual environment| XGBoost | 0.96+ | 0.80+ | 90%+ | 75%+ |

```| Random Forest | 0.95+ | 0.78+ | 90%+ | 72%+ |

| Logistic Regression | 0.93+ | 0.72+ | 90%+ | 65%+ |

---

---

## 🔬 Model Details

## 🔧 Common Commands

### Algorithm

- **XGBoost Classifier** with optimized hyperparameters```bash

# Train with sample

### Hyperparameterspython src/train_models.py --sample 100000

```python

n_estimators=300# Train without SMOTE

max_depth=12python src/train_models.py --no-smote

learning_rate=0.05

reg_lambda=3.6# Train autoencoder (optional)

reg_alpha=3.6python src/autoencoder.py --sample 100000

subsample=0.8

eval_metric='aucpr'# Start API

```python src/api.py



### Feature Engineering (50 Features)# Test API

python test_api.py

**Base Features:**

- Transaction details (amount, type, category)# Check API health

- Temporal features (hour, day, day_of_week, month)curl http://localhost:8000/health

- Risk scores (velocity, spending deviation, geo anomaly)```



**Engineered Features:**---

- Amount transformations (log, ratios)

- Account network features (degrees, transaction counts)## 📖 API Endpoints

- Fraud aggregations (fraud percentages per account)

- Statistical aggregations (by merchant, device, location)- `GET /` - API info

- Time-based features (gaps, frequency)- `GET /health` - Health check

- Risk indicators (night transactions, weekends, self-transfers)- `GET /models` - List models

- `POST /predict` - Single prediction

### Data Balancing- `POST /predict/batch` - Batch predictions

- **Original**: 96.41% legitimate, 3.59% fraud- `GET /docs` - Interactive documentation

- **Balanced**: 66.67% legitimate, 33.33% fraud (2:1 downsampling)

---

### Performance Metrics

- **ROC-AUC**: 0.9911## 🐛 Troubleshooting

- **Precision**: 95% (fraud class)

- **Recall**: 100% (fraud class)**"File not found" error:**

- **F1-Score**: 97% (fraud class)- Download dataset from Kaggle first

- **Overall Accuracy**: 98%- Place in `data/` folder



---**"Out of memory" error:**

- Use smaller sample: `--sample 100000`

## 🔍 Risk Factors Detected- Close other applications



The model identifies fraud based on:**"Models not loaded" in API:**

- Train models first: `python src/train_models.py --sample 100000`

1. **High Velocity Score** (>15): Rapid consecutive transactions

2. **High Spending Deviation** (>2): Unusual spending patterns**Import errors:**

3. **High Geo Anomaly** (>0.7): Suspicious location changes- Reinstall: `pip install -r requirements.txt`

4. **Large Amounts** (>$1000): High-value transactions

5. **Short Time Gaps** (<300s): Frequent rapid transactions---

6. **Night Transactions** (22:00-06:00): Off-hours activity

7. **Self-Transfers**: Same sender and receiver accounts## 📚 Learn More



---1. **Check model performance:**

   ```bash

## 📋 Requirements   cat models/model_comparison.csv

   ```

### System Requirements

- Python 3.8 or higher2. **View API documentation:**

- 16GB RAM (minimum)   - http://localhost:8000/docs (interactive)

- 2GB free disk space   - http://localhost:8000/redoc (detailed)

- Internet connection (for dataset download)

3. **Customize settings:**

### Python Dependencies   - Edit `src/config.py`

See `requirements.txt` for full list. Key packages:   - Adjust model hyperparameters

- pandas >= 2.0.0   - Change target recall threshold

- numpy >= 1.24.0

- scikit-learn >= 1.3.0---

- xgboost >= 2.0.0

- fastapi >= 0.100.0## 🎓 For Beginners

- uvicorn >= 0.20.0

This project demonstrates:

---- ✅ Data preprocessing & feature engineering

- ✅ Handling imbalanced datasets (SMOTE)

## 🛠️ Troubleshooting- ✅ Training multiple ML models

- ✅ Model evaluation & comparison

### Model Not Loading- ✅ Threshold optimization

```bash- ✅ Building REST APIs with FastAPI

# Retrain the model- ✅ Production-ready fraud detection

python train.py

```**Learning path:**

1. Understand the data (`data/README.md`)

### API Not Starting2. Review preprocessing (`src/data_prep.py`)

```bash3. Study model training (`src/train_models.py`)

# Check if port 8000 is in use4. Explore API code (`src/api.py`)

lsof -i :80005. Test predictions (`test_api.py`)



# Kill existing process---

pkill -f uvicorn

## 🛠️ Technology Stack

# Restart API

python3 -m uvicorn api:app --host 0.0.0.0 --port 8000- Python 3.10+

```- scikit-learn (ML models)

- XGBoost (Gradient boosting)

### Memory Issues- imbalanced-learn (SMOTE)

- Reduce dataset size in `train.py` by setting `sample_size` parameter- FastAPI (API server)

- Close other applications- TensorFlow/Keras (Autoencoder)

- Increase system swap space- pandas, numpy (Data processing)



------



## 📝 License## 📝 License



This project is for educational and research purposes.MIT License - Free for personal and commercial use



------



## 🤝 Contributing## 🙏 Acknowledgments



Contributions are welcome! Please feel free to submit a Pull Request.- **Dataset**: [Kaggle Financial Transactions Dataset](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection)

- **Author**: [@mdmakky](https://github.com/mdmakky)

---

---

## 📧 Contact

**⭐ Star this repo if you find it helpful!**

For questions or issues, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- Dataset: [Kaggle Financial Fraud Detection Dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)
- Libraries: XGBoost, scikit-learn, FastAPI, pandas

---

**Built with ❤️ for secure financial transactions**
