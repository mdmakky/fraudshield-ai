"""
FastAPI service for real-time fraud detection.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FraudShield AI",
    description="Real-time credit card fraud detection API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables (loaded once at startup)
MODEL = None
SCALER = None
THRESHOLD = 0.5
FEATURE_NAMES = None


class Transaction(BaseModel):
    """Transaction data model."""
    Time: float = Field(..., description="Seconds elapsed from first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0,
                "Amount": 150.00,
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.53634673796914,
                "V4": 1.37815522427443,
                "V5": -0.338320769942518,
                "V6": 0.462387777762292,
                "V7": 0.239598554061257,
                "V8": 0.0986979012610507,
                "V9": 0.363786969611213,
                "V10": 0.0907941719789316,
                "V11": -0.551599533260813,
                "V12": -0.617800855762348,
                "V13": -0.991389847235408,
                "V14": -0.311169353699879,
                "V15": 1.46817697209427,
                "V16": -0.470400525259478,
                "V17": 0.207971241929242,
                "V18": 0.0257905801985591,
                "V19": 0.403992960255733,
                "V20": 0.251412098239705,
                "V21": -0.018306777944153,
                "V22": 0.277837575558899,
                "V23": -0.110473910188767,
                "V24": 0.0669280749146731,
                "V25": 0.128539358273528,
                "V26": -0.189114843888824,
                "V27": 0.133558376740387,
                "V28": -0.0210530534538215
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model."""
    fraud_score: float = Field(..., description="Fraud probability (0-1)")
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    risk_level: str = Field(..., description="Risk level: low/medium/high/critical")
    threshold: float = Field(..., description="Decision threshold used")
    message: str = Field(..., description="Human-readable message")


class BatchTransaction(BaseModel):
    """Batch of transactions for bulk prediction."""
    transactions: List[Transaction]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_processed: int
    fraud_count: int


def load_models():
    """Load trained models and scaler at startup."""
    global MODEL, SCALER, THRESHOLD, FEATURE_NAMES
    
    try:
        MODEL = joblib.load('models/xgb.joblib')
        SCALER = joblib.load('models/scaler.joblib')
        
        # Try to load optimal threshold from model comparison
        try:
            comparison = pd.read_csv('models/model_comparison.csv')
            xgb_row = comparison[comparison['Model'] == 'XGBoost']
            if not xgb_row.empty:
                THRESHOLD = float(xgb_row['Optimal Threshold'].values[0])
        except:
            THRESHOLD = 0.3  # Default conservative threshold
        
        # Define expected feature names
        FEATURE_NAMES = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 
                        'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 
                        'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 
                        'V25', 'V26', 'V27', 'V28', 'Amount', 'Amount_log', 
                        'Hour', 'Day', 'Amount_bin']
        
        logger.info(f"Models loaded successfully. Threshold: {THRESHOLD}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def preprocess_transaction(txn: Transaction) -> pd.DataFrame:
    """
    Preprocess transaction with feature engineering.
    
    Args:
        txn: Transaction object
    
    Returns:
        DataFrame with engineered features
    """
    # Convert to dict and then DataFrame
    data = txn.model_dump()
    df = pd.DataFrame([data])
    
    # Feature engineering (same as training)
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Day'] = df['Time'] // (3600 * 24)
    
    # Amount binning (use quantiles from training if available, else simple bins)
    if df['Amount'].values[0] == 0:
        df['Amount_bin'] = 0
    elif df['Amount'].values[0] < 10:
        df['Amount_bin'] = 0
    elif df['Amount'].values[0] < 50:
        df['Amount_bin'] = 1
    elif df['Amount'].values[0] < 150:
        df['Amount_bin'] = 2
    elif df['Amount'].values[0] < 500:
        df['Amount_bin'] = 3
    else:
        df['Amount_bin'] = 4
    
    # Scale features
    cols_to_scale = ['Amount', 'Amount_log', 'Hour', 'Day', 'Amount_bin']
    df[cols_to_scale] = SCALER.transform(df[cols_to_scale])
    
    # Ensure correct column order
    df = df[FEATURE_NAMES]
    
    return df


def determine_risk_level(score: float) -> str:
    """
    Determine risk level based on fraud score.
    
    Args:
        score: Fraud probability
    
    Returns:
        Risk level string
    """
    if score < 0.2:
        return "low"
    elif score < 0.5:
        return "medium"
    elif score < 0.8:
        return "high"
    else:
        return "critical"


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting FraudShield AI API...")
    load_models()
    logger.info("API ready to serve predictions")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "FraudShield AI",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "scaler_loaded": SCALER is not None,
        "threshold": THRESHOLD
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Predict fraud for a single transaction.
    
    Args:
        transaction: Transaction data
    
    Returns:
        Prediction response with fraud score and classification
    """
    try:
        # Preprocess transaction
        df = preprocess_transaction(transaction)
        
        # Get prediction probability
        fraud_prob = float(MODEL.predict_proba(df)[0, 1])
        
        # Make binary prediction
        is_fraud = bool(fraud_prob >= THRESHOLD)
        
        # Determine risk level
        risk_level = determine_risk_level(fraud_prob)
        
        # Create message
        if is_fraud:
            message = f"⚠️ FRAUD ALERT: High risk detected (score: {fraud_prob:.2%})"
        else:
            message = f"✓ Transaction appears legitimate (score: {fraud_prob:.2%})"
        
        logger.info(f"Prediction: score={fraud_prob:.4f}, fraud={is_fraud}, risk={risk_level}")
        
        return PredictionResponse(
            fraud_score=fraud_prob,
            is_fraud=is_fraud,
            risk_level=risk_level,
            threshold=THRESHOLD,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(batch: BatchTransaction):
    """
    Predict fraud for multiple transactions.
    
    Args:
        batch: Batch of transactions
    
    Returns:
        Batch prediction response
    """
    try:
        predictions = []
        
        for txn in batch.transactions:
            result = await predict(txn)
            predictions.append(result)
        
        fraud_count = sum(1 for p in predictions if p.is_fraud)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            fraud_count=fraud_count
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/update-threshold")
async def update_threshold(new_threshold: float):
    """
    Update the decision threshold.
    
    Args:
        new_threshold: New threshold value (0-1)
    
    Returns:
        Confirmation message
    """
    global THRESHOLD
    
    if not 0 <= new_threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
    
    old_threshold = THRESHOLD
    THRESHOLD = new_threshold
    
    logger.info(f"Threshold updated: {old_threshold} -> {THRESHOLD}")
    
    return {
        "message": "Threshold updated successfully",
        "old_threshold": old_threshold,
        "new_threshold": THRESHOLD
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
