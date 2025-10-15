"""
SHAP-based model explainability for fraud detection.
Explains individual predictions and feature importance.
"""
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_prep import load_data, feature_engineer, split_scale


def explain_model_global(model, X_sample, feature_names, model_name='XGBoost',
                         max_display=20, save_path='models/shap_summary.png'):
    """
    Generate global SHAP explanations (feature importance across dataset).
    
    Args:
        model: Trained model
        X_sample: Sample of data for explanation
        feature_names: List of feature names
        model_name: Name of the model
        max_display: Maximum features to display
        save_path: Path to save plot
    """
    print(f"\nGenerating SHAP explanations for {model_name}...")
    
    # Create SHAP explainer (TreeExplainer for tree-based models)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except:
        # Fallback to KernelExplainer for non-tree models
        print("Using KernelExplainer (slower)...")
        explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get SHAP values for fraud class
    
    # Summary plot (global feature importance)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_sample, 
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ SHAP summary plot saved to {save_path}")
    plt.close()
    
    # Bar plot (mean absolute SHAP values)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type='bar',
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    bar_path = save_path.replace('.png', '_bar.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"✓ SHAP bar plot saved to {bar_path}")
    plt.close()
    
    return explainer, shap_values


def explain_prediction(explainer, X_instance, feature_names, 
                       instance_idx=0, save_path='models/shap_force.png'):
    """
    Explain a single prediction using force plot.
    
    Args:
        explainer: SHAP explainer
        X_instance: Single instance to explain
        feature_names: List of feature names
        instance_idx: Index of instance (for labeling)
        save_path: Path to save plot
    """
    shap_values = explainer.shap_values(X_instance)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Fraud class
    
    # Force plot
    plt.figure(figsize=(20, 3))
    shap.force_plot(
        explainer.expected_value if not isinstance(explainer.expected_value, list) 
        else explainer.expected_value[1],
        shap_values[0] if len(shap_values.shape) > 1 else shap_values,
        X_instance.iloc[0] if hasattr(X_instance, 'iloc') else X_instance[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ SHAP force plot for instance {instance_idx} saved to {save_path}")
    plt.close()


def explain_top_predictions(model, explainer, X_test, y_test, feature_names,
                           n_examples=5, save_dir='models'):
    """
    Explain top fraud predictions.
    
    Args:
        model: Trained model
        explainer: SHAP explainer
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        n_examples: Number of examples to explain
        save_dir: Directory to save plots
    """
    # Get predictions
    probs = model.predict_proba(X_test)[:, 1]
    
    # Get top fraud predictions
    top_fraud_indices = np.argsort(probs)[-n_examples:][::-1]
    
    print(f"\nExplaining top {n_examples} fraud predictions...")
    
    for i, idx in enumerate(top_fraud_indices):
        actual_label = 'FRAUD' if y_test.iloc[idx] == 1 else 'LEGITIMATE'
        prob = probs[idx]
        
        print(f"\nInstance {i+1}: Index={idx}, Prob={prob:.4f}, Actual={actual_label}")
        
        X_instance = X_test.iloc[[idx]]
        save_path = f'{save_dir}/shap_force_top{i+1}_idx{idx}.png'
        explain_prediction(explainer, X_instance, feature_names, idx, save_path)
    
    # Also explain some false positives if any
    false_positives = np.where((probs > 0.5) & (y_test == 0))[0]
    
    if len(false_positives) > 0:
        print(f"\nExplaining {min(3, len(false_positives))} false positives...")
        for i, idx in enumerate(false_positives[:3]):
            prob = probs[idx]
            print(f"\nFalse Positive {i+1}: Index={idx}, Prob={prob:.4f}")
            
            X_instance = X_test.iloc[[idx]]
            save_path = f'{save_dir}/shap_force_fp{i+1}_idx{idx}.png'
            explain_prediction(explainer, X_instance, feature_names, idx, save_path)


def feature_dependence_plots(shap_values, X_sample, feature_names, 
                             top_n=5, save_dir='models'):
    """
    Create dependence plots for top features.
    
    Args:
        shap_values: SHAP values
        X_sample: Sample data
        feature_names: Feature names
        top_n: Number of top features to plot
        save_dir: Directory to save plots
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_features_idx]
    
    print(f"\nCreating dependence plots for top {top_n} features...")
    
    for i, (feat_idx, feat_name) in enumerate(zip(top_features_idx, top_features)):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        save_path = f'{save_dir}/shap_dependence_{feat_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dependence plot for '{feat_name}' saved")
        plt.close()


def main():
    """Main SHAP explanation pipeline."""
    print("="*60)
    print("  SHAP MODEL EXPLAINABILITY")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_data('data/creditcard.csv')
    df = feature_engineer(df)
    X_train, X_test, y_train, y_test, scaler = split_scale(df)
    
    feature_names = list(X_train.columns)
    
    # Load trained models
    print("\nLoading trained models...")
    models = {
        'XGBoost': joblib.load('models/xgb.joblib'),
        'RandomForest': joblib.load('models/rf.joblib'),
        'LogisticRegression': joblib.load('models/lr.joblib')
    }
    
    # Use a sample for SHAP (for speed)
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    y_sample = y_test.loc[X_sample.index]
    
    print(f"Using sample of {sample_size} instances for SHAP analysis")
    
    # Explain each model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"  Explaining {model_name}")
        print(f"{'='*60}")
        
        save_path = f'models/shap_summary_{model_name.lower()}.png'
        
        # Global explanations
        explainer, shap_values = explain_model_global(
            model, X_sample, feature_names, 
            model_name=model_name,
            save_path=save_path
        )
        
        # Explain top predictions (only for XGBoost to save time)
        if model_name == 'XGBoost':
            explain_top_predictions(
                model, explainer, X_sample, y_sample, 
                feature_names, n_examples=5
            )
            
            # Dependence plots for top features
            feature_dependence_plots(
                shap_values, X_sample, feature_names, top_n=5
            )
    
    print("\n" + "="*60)
    print("  SHAP ANALYSIS COMPLETE!")
    print("="*60)
    print("\nAll SHAP plots saved to models/ directory")


if __name__ == "__main__":
    main()
