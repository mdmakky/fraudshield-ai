"""
Autoencoder-based anomaly detection for fraud detection.
Unsupervised approach: train on legitimate transactions only.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from data_prep import load_data, feature_engineer, split_scale


def build_autoencoder(n_features, latent_dim=8, hidden_layers=[64, 32]):
    """
    Build autoencoder model for anomaly detection.
    
    Args:
        n_features: Number of input features
        latent_dim: Size of bottleneck layer
        hidden_layers: List of hidden layer sizes
    
    Returns:
        Compiled Keras model
    """
    # Encoder
    inp = layers.Input(shape=(n_features,))
    x = inp
    
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    
    # Bottleneck
    encoded = layers.Dense(latent_dim, activation='relu', name='bottleneck')(x)
    
    # Decoder (mirror of encoder)
    x = encoded
    for units in reversed(hidden_layers):
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    
    # Output layer
    decoded = layers.Dense(n_features, activation='linear')(x)
    
    # Build model
    autoencoder = models.Model(inp, decoded, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return autoencoder


def train_autoencoder(X_train_normal, n_features, epochs=50, batch_size=256, 
                      validation_split=0.1, latent_dim=8):
    """
    Train autoencoder on legitimate transactions only.
    
    Args:
        X_train_normal: Training data (only legitimate transactions)
        n_features: Number of features
        epochs: Training epochs
        batch_size: Batch size
        validation_split: Validation split ratio
        latent_dim: Bottleneck dimension
    
    Returns:
        Trained autoencoder model
    """
    print(f"Building autoencoder with {n_features} features, latent dim={latent_dim}")
    
    ae = build_autoencoder(n_features, latent_dim=latent_dim)
    ae.summary()
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    print("\nTraining autoencoder...")
    history = ae.fit(
        X_train_normal, X_train_normal,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1
    )
    
    return ae, history


def compute_reconstruction_error(model, X):
    """
    Compute reconstruction error (anomaly score).
    
    Args:
        model: Trained autoencoder
        X: Input data
    
    Returns:
        Array of reconstruction errors (MSE per sample)
    """
    reconstructed = model.predict(X, verbose=0)
    mse = np.mean(np.square(X - reconstructed), axis=1)
    return mse


def evaluate_autoencoder(ae, X_test, y_test, threshold=None):
    """
    Evaluate autoencoder performance on test set.
    
    Args:
        ae: Trained autoencoder
        X_test: Test features
        y_test: Test labels
        threshold: Anomaly threshold (if None, will be determined)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Compute reconstruction errors
    errors = compute_reconstruction_error(ae, X_test)
    
    # If no threshold provided, use precision-recall curve
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(y_test, errors)
        
        # Find threshold at recall >= 0.90
        indices = np.where(recall >= 0.90)[0]
        if len(indices) > 0:
            best_idx = indices[np.argmax(precision[indices])]
            threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[0]
        else:
            threshold = np.percentile(errors, 95)  # Default to 95th percentile
        
        print(f"Optimal threshold: {threshold:.6f}")
    
    # Make predictions
    predictions = (errors > threshold).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\nAutoencoder Evaluation:")
    print(classification_report(y_test, predictions, target_names=['Legitimate', 'Fraud']))
    
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    roc_auc = roc_auc_score(y_test, errors)
    pr_auc = average_precision_score(y_test, errors)
    
    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    return {
        'threshold': threshold,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'errors': errors,
        'predictions': predictions
    }


def plot_reconstruction_error_distribution(errors_normal, errors_fraud, 
                                           threshold, save_path='models/ae_error_dist.png'):
    """Plot distribution of reconstruction errors."""
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(errors_normal, bins=50, alpha=0.7, label='Legitimate', density=True)
    plt.hist(errors_fraud, bins=50, alpha=0.7, label='Fraud', density=True)
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold:.4f}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.yscale('log')
    
    # Box plot
    plt.subplot(1, 2, 2)
    data = [errors_normal, errors_fraud]
    plt.boxplot(data, labels=['Legitimate', 'Fraud'])
    plt.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold:.4f}')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Reconstruction Error by Class')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nError distribution plot saved to {save_path}")
    plt.close()


def plot_training_history(history, save_path='models/ae_training_history.png'):
    """Plot training history."""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History - Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training History - MAE')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training history saved to {save_path}")
    plt.close()


def main():
    """Main autoencoder training pipeline."""
    print("="*60)
    print("  AUTOENCODER ANOMALY DETECTION")
    print("="*60)
    
    # Load and prepare data
    print("\nLoading data...")
    df = load_data('data/creditcard.csv')
    df = feature_engineer(df)
    X_train, X_test, y_train, y_test, scaler = split_scale(df)
    
    # Extract only legitimate transactions for training
    print("\nExtracting legitimate transactions for training...")
    X_train_normal = X_train[y_train == 0].values
    print(f"Training on {len(X_train_normal)} legitimate transactions")
    
    # Train autoencoder
    n_features = X_train_normal.shape[1]
    ae, history = train_autoencoder(
        X_train_normal,
        n_features=n_features,
        epochs=50,
        batch_size=256,
        latent_dim=8
    )
    
    # Save model
    ae.save('models/autoencoder.h5')
    print("\n✓ Autoencoder saved to models/autoencoder.h5")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("  EVALUATION ON TEST SET")
    print("="*60)
    
    results = evaluate_autoencoder(ae, X_test.values, y_test)
    
    # Save threshold
    threshold_info = {
        'threshold': results['threshold'],
        'roc_auc': results['roc_auc'],
        'pr_auc': results['pr_auc']
    }
    joblib.dump(threshold_info, 'models/ae_threshold.joblib')
    print("\n✓ Threshold info saved to models/ae_threshold.joblib")
    
    # Plot error distributions
    errors_test = results['errors']
    errors_normal = errors_test[y_test == 0]
    errors_fraud = errors_test[y_test == 1]
    
    plot_reconstruction_error_distribution(
        errors_normal, errors_fraud, results['threshold']
    )
    
    print("\n" + "="*60)
    print("  AUTOENCODER TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
