from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

app = FastAPI(title="FraudShield AI", description="High-Accuracy Fraud Detection API")

# Load model and preprocessing objects
try:
    model = joblib.load('models/fraud_detection_model.joblib')
    label_encoders = joblib.load('models/label_encoders.joblib')
    feature_cols = joblib.load('models/feature_columns.joblib')
    print("✓ Model and preprocessing objects loaded successfully")
except FileNotFoundError:
    print("⚠️  Model files not found. Please run training first.")
    model = None

class Transaction(BaseModel):
    timestamp: str
    sender_account: str
    receiver_account: str
    amount: float
    transaction_type: str
    merchant_category: str
    location: str
    device_used: str
    payment_channel: str
    time_since_last_transaction: float
    spending_deviation_score: float
    velocity_score: float
    geo_anomaly_score: float

def preprocess_transaction(transaction: Transaction):
    """Preprocess a single transaction for prediction."""
    # Create dataframe
    df = pd.DataFrame([transaction.dict()])

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Convert boolean to int (though not used here)
    # df["is_fraud"] = df["is_fraud"].astype(int)  # Not needed for prediction

    # Categorical encoding
    categorical_cols = ['transaction_type', 'merchant_category', 'location', 'device_used', 'payment_channel']
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            try:
                df[f'{col}_encoded'] = label_encoders[col].transform(df[col].astype(str))
            except ValueError:
                # Handle unseen categories
                df[f'{col}_encoded'] = 0

    # Amount features
    df["amount_per_velocity"] = df["amount"] / (df["velocity_score"] + 1)
    df["amount_log"] = np.log1p(df["amount"])

    # For single transaction, set more realistic default values based on transaction characteristics
    # These are approximations since we can't compute real aggregations
    velocity_high = (df["velocity_score"] > 15).any()
    deviation_high = (df["spending_deviation_score"] > 2).any()
    geo_high = (df["geo_anomaly_score"] > 0.7).any()
    amount_high = (df["amount"] > 1000).any()
    time_low = (df["time_since_last_transaction"] < 300).any()
    night_time = (df["hour"] < 6).any() or (df["hour"] > 22).any()
    self_transfer = (df["sender_account"] == df["receiver_account"]).any()

    # More aggressive risk scoring
    risk_score = sum([velocity_high, deviation_high, geo_high, amount_high, time_low, night_time, self_transfer])

    if risk_score >= 3:  # High risk if 3+ risk factors
        high_risk = True
    elif risk_score >= 1:  # Medium risk
        high_risk = True  # Still treat as high for conservative detection
    else:
        high_risk = False

    df["amount_to_avg_ratio"] = 1.5 if high_risk else 0.8
    df["transaction_per_day"] = 5 if high_risk else 1
    df["transaction_gap"] = 60 if high_risk else 3600

    # Risk features
    df["is_night_transaction"] = df["hour"].between(18, 24).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_self_transfer"] = (df["sender_account"] == df["receiver_account"]).astype(int)

    # Network features (set based on risk)
    df["sender_degree"] = 10 if high_risk else 3
    df["receiver_degree"] = 8 if high_risk else 2
    df["sender_total_transaction"] = 20 if high_risk else 5
    df["receiver_total_transaction"] = 15 if high_risk else 3

    # Aggregation features (set based on risk - make more extreme)
    if high_risk:
        df["sender_avg_amount"] = df["amount"] * 2.5
        df["sender_std_amount"] = df["amount"] * 0.8
        df["sender_fraud_transaction"] = 8
        df["receiver_fraud_transaction"] = 6
        df["sender_fraud_percentage"] = 40.0
        df["receiver_fraud_percentage"] = 35.0
    else:
        df["sender_avg_amount"] = df["amount"] * 0.7
        df["sender_std_amount"] = df["amount"] * 0.2
        df["sender_fraud_transaction"] = 0
        df["receiver_fraud_transaction"] = 0
        df["sender_fraud_percentage"] = 0.0
        df["receiver_fraud_percentage"] = 0.0

    # Others
    df["deviation_squared"] = df["spending_deviation_score"] ** 2

    # Statistical aggregations (set based on risk)
    df["amount_mean_merchant"] = df["amount"] * (1.2 if high_risk else 0.9)
    df["amount_std_merchant"] = df["amount"] * 0.4
    df["amount_min_merchant"] = df["amount"] * 0.5
    df["amount_max_merchant"] = df["amount"] * 2.0
    df["velocity_score_mean_merchant"] = df["velocity_score"] * (1.1 if high_risk else 0.9)
    df["velocity_score_std_merchant"] = df["velocity_score"] * 0.2
    df["is_fraud_mean_merchant"] = 0.1 if high_risk else 0.01

    df["amount_mean_device"] = df["amount"] * (1.1 if high_risk else 0.95)
    df["amount_std_device"] = df["amount"] * 0.3
    df["velocity_score_mean_device"] = df["velocity_score"] * (1.05 if high_risk else 0.95)
    df["velocity_score_std_device"] = df["velocity_score"] * 0.15
    df["is_fraud_mean_device"] = 0.08 if high_risk else 0.005

    df["amount_mean_location"] = df["amount"] * (1.3 if high_risk else 0.85)
    df["amount_std_location"] = df["amount"] * 0.5
    df["geo_anomaly_score_mean_location"] = df["geo_anomaly_score"] * (1.2 if high_risk else 0.8)
    df["geo_anomaly_score_std_location"] = df["geo_anomaly_score"] * 0.3
    df["is_fraud_mean_location"] = 0.12 if high_risk else 0.008

    # Drop original columns
    cols_to_drop = ['timestamp', 'sender_account', 'receiver_account'] + categorical_cols
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Ensure all required features are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[feature_cols]

    return df.values  # Return numpy array for prediction

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    """Predict if a transaction is fraudulent."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")

    try:
        # Preprocess transaction
        features = preprocess_transaction(transaction)

        # Make prediction
        fraud_probability = model.predict_proba(features)[0, 1]
        is_fraud = fraud_probability > 0.5

        return {
            "is_fraud": bool(is_fraud),
            "fraud_probability": float(fraud_probability),
            "confidence": "high" if fraud_probability > 0.8 else "medium" if fraud_probability > 0.6 else "low"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/")
def root():
    """API root endpoint."""
    return {"message": "FraudShield AI - High Accuracy Fraud Detection API"}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)