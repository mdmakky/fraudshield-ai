# Example: Test the FraudShield API

import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

# Example transaction (legitimate)
legitimate_transaction = {
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

# Example transaction (suspicious - high amount + unusual V values)
suspicious_transaction = {
    "Time": 82500.0,
    "Amount": 5000.00,
    "V1": -3.0,
    "V2": 1.5,
    "V3": -4.0,
    "V4": 2.8,
    "V5": -2.5,
    "V6": 1.2,
    "V7": -3.5,
    "V8": 2.1,
    "V9": -1.8,
    "V10": 3.2,
    "V11": -2.0,
    "V12": -4.5,
    "V13": 3.8,
    "V14": -5.2,
    "V15": 2.9,
    "V16": -1.5,
    "V17": -3.8,
    "V18": 1.1,
    "V19": -2.3,
    "V20": 1.9,
    "V21": -0.8,
    "V22": 2.5,
    "V23": -1.2,
    "V24": 0.9,
    "V25": -2.1,
    "V26": 1.7,
    "V27": -0.5,
    "V28": 0.3
}


def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_prediction(transaction, label):
    """Test single prediction."""
    print(f"Testing {label} transaction...")
    response = requests.post(
        f"{BASE_URL}/predict",
        json=transaction
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response:")
    print(f"  Fraud Score: {result['fraud_score']:.4f}")
    print(f"  Is Fraud: {result['is_fraud']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Message: {result['message']}\n")


def test_batch_prediction():
    """Test batch prediction."""
    print("Testing batch prediction...")
    batch = {
        "transactions": [legitimate_transaction, suspicious_transaction]
    }
    response = requests.post(
        f"{BASE_URL}/batch-predict",
        json=batch
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total Processed: {result['total_processed']}")
    print(f"Fraud Count: {result['fraud_count']}\n")
    
    for i, pred in enumerate(result['predictions']):
        print(f"Transaction {i+1}:")
        print(f"  Score: {pred['fraud_score']:.4f}, Fraud: {pred['is_fraud']}, Risk: {pred['risk_level']}")


if __name__ == "__main__":
    print("="*60)
    print("FraudShield AI - API Test")
    print("="*60)
    print()
    
    try:
        # Test endpoints
        test_health()
        test_prediction(legitimate_transaction, "legitimate")
        test_prediction(suspicious_transaction, "suspicious")
        test_batch_prediction()
        
        print("="*60)
        print("All tests completed successfully! ✓")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API.")
        print("Make sure the API is running:")
        print("  uvicorn src.api:app --reload")
    except Exception as e:
        print(f"ERROR: {e}")
