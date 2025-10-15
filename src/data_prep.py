"""
Data preparation and feature engineering for credit card fraud detection.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


def load_data(path='data/creditcard.csv'):
    """Load credit card dataset from CSV."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} transactions")
    print(f"Fraud cases: {df['Class'].sum()} ({100*df['Class'].mean():.2f}%)")
    return df


def feature_engineer(df):
    """
    Create additional features from raw data.
    
    The Kaggle dataset has:
    - V1 to V28: PCA-transformed features
    - Time: seconds elapsed from first transaction
    - Amount: transaction amount
    - Class: 0=legitimate, 1=fraud
    """
    df = df.copy()
    
    # Log-transform amount (handles skewness and zeros)
    df['Amount_log'] = np.log1p(df['Amount'])
    
    # Extract hour of day from Time (assuming Time is seconds from start)
    df['Hour'] = (df['Time'] // 3600) % 24
    
    # Create time-based features
    df['Day'] = df['Time'] // (3600 * 24)
    
    # Amount bins (categorical encoding of amount ranges)
    df['Amount_bin'] = pd.qcut(df['Amount'], q=5, labels=False, duplicates='drop')
    
    print(f"Feature engineering complete. Shape: {df.shape}")
    return df


def split_scale(df, test_size=0.2, random_state=42):
    """
    Split data into train/test and apply scaling.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Stratified split to preserve fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"Train set: {len(X_train)} samples, {y_train.sum()} frauds")
    print(f"Test set: {len(X_test)} samples, {y_test.sum()} frauds")
    
    # Scale specific columns with RobustScaler (less sensitive to outliers)
    scaler = RobustScaler()
    cols_to_scale = ['Amount', 'Amount_log', 'Hour', 'Day', 'Amount_bin']
    
    # Fit only on training data
    scaler.fit(X_train[cols_to_scale])
    
    # Transform both train and test
    X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Example usage
    df = load_data()
    df = feature_engineer(df)
    X_train, X_test, y_train, y_test, scaler = split_scale(df)
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
