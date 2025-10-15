"""
Train multiple fraud detection models with imbalance handling.
Includes Logistic Regression, Random Forest, and XGBoost.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, 
    precision_recall_curve, 
    roc_auc_score, 
    confusion_matrix,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

from data_prep import load_data, feature_engineer, split_scale


def evaluate_model(model, X, y, name='Model', threshold=0.5):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        model: Trained classifier
        X: Features
        y: True labels
        name: Model name for display
        threshold: Decision threshold
    
    Returns:
        dict with evaluation metrics
    """
    # Get probability predictions
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    print(f"\n{'='*60}")
    print(f"  {name} Evaluation (threshold={threshold})")
    print(f"{'='*60}")
    
    # Classification report
    print(classification_report(y, preds, target_names=['Legitimate', 'Fraud']))
    
    # ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score: {pr_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, preds)
    print("\nConfusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives: {tn}, False Positives: {fp}")
    print(f"False Negatives: {fn}, True Positives: {tp}")
    
    return {
        'name': name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'probs': probs,
        'threshold': threshold
    }


def find_optimal_threshold(y_true, y_probs, target_recall=0.90):
    """
    Find optimal threshold for a target recall level.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        target_recall: Minimum desired recall
    
    Returns:
        optimal threshold and corresponding precision
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find thresholds where recall >= target
    indices = np.where(recall >= target_recall)[0]
    
    if len(indices) == 0:
        print(f"Warning: Cannot achieve recall >= {target_recall}")
        return thresholds[0], precision[0], recall[0]
    
    # Among those, pick the one with highest precision
    best_idx = indices[np.argmax(precision[indices])]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    
    print(f"\nOptimal Threshold for Recall >= {target_recall}:")
    print(f"  Threshold: {best_threshold:.4f}")
    print(f"  Precision: {best_precision:.4f}")
    print(f"  Recall: {best_recall:.4f}")
    
    return best_threshold, best_precision, best_recall


def plot_precision_recall_curve(results, y_test, save_path='models/pr_curve.png'):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(10, 6))
    
    for result in results:
        precision, recall, _ = precision_recall_curve(y_test, result['probs'])
        plt.plot(recall, precision, label=f"{result['name']} (AUC={result['pr_auc']:.3f})")
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPR curve saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline."""
    print("Loading and preparing data...")
    df = load_data('data/creditcard.csv')
    df = feature_engineer(df)
    X_train, X_test, y_train, y_test, scaler = split_scale(df)
    
    # Handle class imbalance with SMOTE
    print("\nApplying SMOTE to balance training data...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"Resampled training set: {len(X_res)} samples")
    print(f"  Legitimate: {(y_res==0).sum()}")
    print(f"  Fraud: {(y_res==1).sum()}")
    
    # ========== Train Models ==========
    
    print("\n" + "="*60)
    print("  TRAINING MODELS")
    print("="*60)
    
    # 1. Logistic Regression
    print("\n[1/3] Training Logistic Regression...")
    lr = LogisticRegression(
        class_weight='balanced', 
        max_iter=1000, 
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_res, y_res)
    joblib.dump(lr, 'models/lr.joblib')
    print("✓ Logistic Regression saved to models/lr.joblib")
    
    # 2. Random Forest
    print("\n[2/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, 
        class_weight='balanced', 
        max_depth=10,
        min_samples_split=10,
        n_jobs=-1, 
        random_state=42,
        verbose=0
    )
    rf.fit(X_res, y_res)
    joblib.dump(rf, 'models/rf.joblib')
    print("✓ Random Forest saved to models/rf.joblib")
    
    # 3. XGBoost
    print("\n[3/3] Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    
    xgb_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X_res, y_res)
    joblib.dump(xgb_clf, 'models/xgb.joblib')
    print("✓ XGBoost saved to models/xgb.joblib")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    print("\n✓ Scaler saved to models/scaler.joblib")
    
    # ========== Evaluate Models ==========
    
    print("\n" + "="*60)
    print("  EVALUATING MODELS ON TEST SET")
    print("="*60)
    
    results = []
    
    # Evaluate with default threshold
    for model, name in [(lr, 'Logistic Regression'), 
                        (rf, 'Random Forest'), 
                        (xgb_clf, 'XGBoost')]:
        result = evaluate_model(model, X_test, y_test, name, threshold=0.5)
        results.append(result)
    
    # Find optimal thresholds for each model
    print("\n" + "="*60)
    print("  THRESHOLD TUNING FOR HIGH RECALL")
    print("="*60)
    
    for result in results:
        print(f"\n{result['name']}:")
        threshold, prec, rec = find_optimal_threshold(
            y_test, result['probs'], target_recall=0.90
        )
        result['optimal_threshold'] = threshold
        result['optimal_precision'] = prec
        result['optimal_recall'] = rec
    
    # Re-evaluate with optimal thresholds
    print("\n" + "="*60)
    print("  RE-EVALUATION WITH OPTIMAL THRESHOLDS")
    print("="*60)
    
    for model, result in zip([lr, rf, xgb_clf], results):
        evaluate_model(
            model, X_test, y_test, 
            name=result['name'], 
            threshold=result['optimal_threshold']
        )
    
    # Plot PR curves
    plot_precision_recall_curve(results, y_test)
    
    # Save results summary
    summary = pd.DataFrame([{
        'Model': r['name'],
        'ROC-AUC': r['roc_auc'],
        'PR-AUC': r['pr_auc'],
        'Default Threshold': r['threshold'],
        'Optimal Threshold': r['optimal_threshold'],
        'Optimal Precision': r['optimal_precision'],
        'Optimal Recall': r['optimal_recall']
    } for r in results])
    
    summary.to_csv('models/model_comparison.csv', index=False)
    print("\n✓ Model comparison saved to models/model_comparison.csv")
    print("\n" + summary.to_string(index=False))
    
    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("="*60)
    print("\nBest model by PR-AUC:", summary.loc[summary['PR-AUC'].idxmax(), 'Model'])


if __name__ == "__main__":
    main()
