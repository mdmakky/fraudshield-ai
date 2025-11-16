import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath, sample_size=None):
    """Load and preprocess the financial transaction data."""
    print("Loading data...")

    if sample_size:
        df = pd.read_csv(filepath, nrows=sample_size)
    else:
        df = pd.read_csv(filepath)

    print(f"Loaded {len(df):,} transactions")

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Convert boolean to int
    df["is_fraud"] = df["is_fraud"].astype(int)

    # Drop fraud_type as it's redundant
    if 'fraud_type' in df.columns:
        df = df.drop(columns=['fraud_type'])

    # Handle missing values for time_since_last_transaction
    df['time_since_last_transaction'] = df['time_since_last_transaction'].fillna(df['time_since_last_transaction'].mean())

    # Categorical encoding (keep originals for now)
    categorical_cols = ['transaction_type', 'merchant_category', 'location', 'device_used', 'payment_channel']
    label_encoders = {}

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    print(f"Final shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3%}")

    return df, label_encoders

def create_advanced_features(df):
    """Create advanced features for better fraud detection."""
    df = df.copy()

    # Amount features
    df["amount_per_velocity"] = df["amount"] / (df["velocity_score"] + 1)
    df["amount_log"] = np.log1p(df["amount"])
    df["amount_to_avg_ratio"] = df["amount"] / df.groupby("sender_account")["amount"].transform("mean")

    # Frequency features
    df["transaction_per_day"] = df.groupby(["sender_account", "day"])["amount"].transform("count")
    df["transaction_gap"] = (df.groupby("sender_account")["timestamp"].diff().dt.total_seconds().fillna(0))

    # Risk features
    df["is_night_transaction"] = df["hour"].between(18, 24).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_self_transfer"] = (df["sender_account"] == df["receiver_account"]).astype(int)

    # Network features
    df["sender_degree"] = df.groupby("sender_account")["receiver_account"].transform("nunique")
    df["receiver_degree"] = df.groupby("receiver_account")["sender_account"].transform("nunique")
    df["sender_total_transaction"] = df.groupby("sender_account")["amount"].transform("count")
    df["receiver_total_transaction"] = df.groupby("receiver_account")["amount"].transform("count")

    # Aggregation features
    df["sender_avg_amount"] = df.groupby("sender_account")["amount"].transform("mean")
    df["sender_std_amount"] = df.groupby("sender_account")["amount"].transform("std").fillna(0)

    # Fraud features
    df["sender_fraud_transaction"] = df.groupby("sender_account")["is_fraud"].transform("sum")
    df["receiver_fraud_transaction"] = df.groupby("receiver_account")["is_fraud"].transform("sum")

    df["sender_fraud_percentage"] = (df["sender_fraud_transaction"] * 100 / df["sender_total_transaction"]).round(2)
    df["receiver_fraud_percentage"] = (df["receiver_fraud_transaction"] * 100 / df["receiver_total_transaction"]).round(2)

    df[["sender_fraud_percentage", "receiver_fraud_percentage"]] = df[["sender_fraud_percentage", "receiver_fraud_percentage"]].fillna(0)

    # Others
    df["deviation_squared"] = df["spending_deviation_score"] ** 2

    # Statistical aggregations by merchant category
    if 'merchant_category_encoded' in df.columns:
        merchant_stats = df.groupby('merchant_category_encoded').agg({
            'amount': ['mean', 'std', 'min', 'max'],
            'velocity_score': ['mean', 'std'],
            'is_fraud': 'mean'
        }).round(4)

        merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns]
        merchant_stats = merchant_stats.reset_index()

        df = df.merge(merchant_stats, on='merchant_category_encoded', how='left', suffixes=('', '_merchant'))

    # Statistical aggregations by device
    if 'device_used_encoded' in df.columns:
        device_stats = df.groupby('device_used_encoded').agg({
            'amount': ['mean', 'std'],
            'velocity_score': ['mean', 'std'],
            'is_fraud': 'mean'
        }).round(4)

        device_stats.columns = ['_'.join(col).strip() for col in device_stats.columns]
        device_stats = device_stats.reset_index()

        df = df.merge(device_stats, on='device_used_encoded', how='left', suffixes=('', '_device'))

    # Statistical aggregations by location
    if 'location_encoded' in df.columns:
        location_stats = df.groupby('location_encoded').agg({
            'amount': ['mean', 'std'],
            'geo_anomaly_score': ['mean', 'std'],
            'is_fraud': 'mean'
        }).round(4)

        location_stats.columns = ['_'.join(col).strip() for col in location_stats.columns]
        location_stats = location_stats.reset_index()

        df = df.merge(location_stats, on='location_encoded', how='left', suffixes=('', '_location'))

    # Fill NaN values from aggregations
    df = df.fillna(df.median(numeric_only=True))

    # Drop account columns and other unnecessary features
    cols_to_drop = ['transaction_id', 'timestamp', 'sender_account', 'receiver_account',
                   'ip_address', 'device_hash', 'transaction_type', 'merchant_category',
                   'location', 'device_used', 'payment_channel']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df

def train_ensemble_model(X_train, y_train, X_test, y_test):
    """Train an ensemble model with advanced techniques for subtle fraud patterns."""

    # Use SMOTE to oversample minority class since patterns are subtle
    smote = SMOTE(random_state=42, sampling_strategy=0.3)  # 30% fraud rate
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: {X_train_smote.shape}, Fraud rate: {y_train_smote.mean():.3%}")

    # Define base models with different approaches
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=0.01,
        scale_pos_weight=1,  # Remove since we're using SMOTE
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        scale_pos_weight=1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    cb_model = cb.CatBoostClassifier(
        iterations=2000,
        depth=6,
        learning_rate=0.01,
        random_strength=0.1,
        bagging_temperature=0.1,
        scale_pos_weight=1,
        random_state=42,
        verbose=False
    )

    # Train individual models with cross-validation
    models = []
    model_names = ['XGBoost', 'LightGBM', 'CatBoost']
    base_models = [xgb_model, lgb_model, cb_model]

    for name, model in zip(model_names, base_models):
        print(f"Training {name}...")
        if name == 'LightGBM':
            model.fit(
                X_train_smote, y_train_smote,
                eval_set=[(X_test, y_test)]
            )
        else:
            model.fit(
                X_train_smote, y_train_smote,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        models.append(model)

    # Create ensemble with weighted voting based on individual performance
    ensemble = VotingClassifier(
        estimators=list(zip(model_names, models)),
        voting='soft'
    )

    print("Training ensemble model...")
    ensemble.fit(X_train_smote, y_train_smote)

    # Predictions
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Evaluation
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nEnsemble Model Results:")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Try different thresholds for better precision/recall balance
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\n=== Threshold Analysis ===")
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba > thresh).astype(int)
        precision = (y_pred_thresh & y_test.astype(int)).sum() / y_pred_thresh.sum() if y_pred_thresh.sum() > 0 else 0
        recall = (y_pred_thresh & y_test.astype(int)).sum() / y_test.sum()
        print(f"Threshold {thresh:.1f}: Precision={precision:.3f}, Recall={recall:.3f}")

    return ensemble, auc_score

def main():
    """Main training pipeline."""
    print("=== FraudShield AI - High Accuracy Training ===")

    # Load and preprocess data
    df, label_encoders = load_and_preprocess_data('data/financial_fraud_detection_dataset.csv')

    # Create advanced features
    df = create_advanced_features(df)

    # Balance the dataset by downsampling majority class to 2:1 ratio
    df_majority = df[df['is_fraud'] == 0]
    df_minority = df[df['is_fraud'] == 1]

    df_majority_downsampled = df_majority.sample(n=2 * len(df_minority), random_state=36)
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    print(f"Balanced dataset shape: {df_balanced.shape}")

    # Prepare features and target
    feature_cols = [col for col in df_balanced.columns if col != 'is_fraud']
    X = df_balanced[feature_cols]
    y = df_balanced['is_fraud']

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Feature names: {feature_cols[:10]}...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=36, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=1/8, random_state=36, stratify=y_train
    )

    print(f"\nTrain shape: {X_train.shape}, Fraud rate: {y_train.mean():.3%}")
    print(f"Test shape: {X_test.shape}, Fraud rate: {y_test.mean():.3%}")
    print(f"Val shape: {X_val.shape}, Fraud rate: {y_val.mean():.3%}")

    # Train XGBoost model
    scale_pos_weight = y_train[y_train == 1].count() / y_train[y_train == 0].count()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        objective='binary:logistic',
        tree_method='hist',
        max_depth=12,
        learning_rate=0.05,
        reg_lambda=3.6,
        reg_alpha=3.6,
        scale_pos_weight=scale_pos_weight,
        eval_metric=['aucpr'],
        verbosity=2,
        subsample=0.8,
        device='cuda' if hasattr(xgb, 'device') else None,  # Use GPU if available
        n_jobs=-1
    )

    print("\nTraining XGBoost...")
    eval_set = [(X_train, y_train), (X_val, y_val)]
    xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=50)

    # Evaluate
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print("\nXGBoost Model Results:")
    print(f"ROC-AUC Score: {auc_score:.4f}")

    # Classification report
    y_pred = (y_pred_proba > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Threshold analysis
    print("\n=== Threshold Analysis ===")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_thresh = (y_pred_proba > thresh).astype(int)
        precision = (y_pred_thresh & y_test).sum() / y_pred_thresh.sum() if y_pred_thresh.sum() > 0 else 0
        recall = (y_pred_thresh & y_test).sum() / y_test.sum()
        print(f"Threshold {thresh:.1f}: Precision={precision:.3f}, Recall={recall:.3f}")

    # Save model and preprocessing objects
    print("\nSaving model and preprocessing objects...")
    joblib.dump(xgb_model, 'models/fraud_detection_model.joblib')
    joblib.dump(label_encoders, 'models/label_encoders.joblib')
    joblib.dump(feature_cols, 'models/feature_columns.joblib')

    print("✓ Model saved successfully!")
    print(f"AUC Score: {auc_score:.4f}")
    if auc_score >= 0.97:
        print("🎉 ACHIEVED TARGET ACCURACY! 🎉")
    else:
        print(f"⚠️  AUC Score: {auc_score:.4f} (Target: 0.97+)")

if __name__ == "__main__":
    main()