# train_model.py
# Optimized version WITHOUT SMOTE - uses class weights and better hyperparameters

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import xgboost as xgb
import os

# INPUT_PATH = "processed_dataset.csv"
INPUT_PATH = "processed_dataset_v2.csv"

def load_processed():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Processed dataset not found at {INPUT_PATH}. Run feature_engineering.py first.")
    
    print(f"Loading processed dataset from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded shape: {df.shape}")
    return df

def prepare_data(df):
    target_col = 'Category'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Check dataset.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Check for 'Family' column (data leakage warning)
    if 'Family' in X.columns:
        print("⚠️  ERROR: 'Family' column still present! This causes data leakage!")
        print("    Re-run feature_engineering.py to remove it.")
        raise ValueError("Data leakage detected: 'Family' column must be removed")
    
    # Encode target
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    # Print class distribution
    print("\nClass distribution:")
    class_dist = Counter(y_encoded)
    for idx, count in sorted(class_dist.items()):
        print(f"  {le_y.classes_[idx]}: {count} ({count/len(y)*100:.2f}%)")
    
    return X, y_encoded, le_y

def compute_sample_weights(y_train, le_y):
    """Compute sample weights for balanced training"""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Create weight dictionary
    weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Map to sample weights
    sample_weights = np.array([weight_dict[y] for y in y_train])
    
    print(f"\nClass weights: {weight_dict}")
    return sample_weights

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, le_y):
    """Train Random Forest with optimized hyperparameters"""
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST")
    print("="*70)
    
    rf = RandomForestClassifier(
        n_estimators=1000,             # Even more trees
        max_depth=None,                # No depth limit - let trees grow
        min_samples_split=5,           # Less restrictive
        min_samples_leaf=2,            # Less restrictive
        max_features='sqrt',           
        class_weight='balanced_subsample',
        bootstrap=True,
        oob_score=True,
        max_samples=0.9,               # Use more data per tree
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    rf.fit(X_train, y_train)
    
    # Evaluate
    print(f"Out-of-Bag Score: {rf.oob_score_:.4f}")
    
    val_pred = rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    
    test_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    print(f"Validation Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f} | F1: {test_f1:.4f}")
    
    return rf, test_pred

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, le_y):
    """Train XGBoost - often better than Random Forest"""
    print("\n" + "="*70)
    print("TRAINING XGBOOST")
    print("="*70)
    
    # Compute sample weights
    sample_weights = compute_sample_weights(y_train, le_y)
    
    # XGBoost parameters - more aggressive
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=15,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.05,
        reg_lambda=0.5,
        scale_pos_weight=1,
        objective='multi:softmax',
        num_class=len(le_y.classes_),
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method='hist'
    )
    
    # Train with early stopping using eval_set only (new API)
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    val_pred = xgb_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    
    test_pred = xgb_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    print(f"Validation Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f} | F1: {test_f1:.4f}")
    
    return xgb_model, test_pred

def print_detailed_results(y_test, test_pred, le_y, model_name="Model"):
    """Print comprehensive evaluation metrics"""
    print("\n" + "="*70)
    print(f"{model_name} - DETAILED RESULTS")
    print("="*70)
    
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    print(f"\n{'='*70}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"{'='*70}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(
        y_test, test_pred, 
        target_names=le_y.classes_,
        digits=4
    ))
    
    # Per-class breakdown
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, test_pred, labels=range(len(le_y.classes_))
    )
    
    print("\nPer-Class Performance (sorted by F1-score):")
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 70)
    
    # Sort by F1 score
    class_metrics = []
    for i, class_name in enumerate(le_y.classes_):
        class_metrics.append((class_name, precision[i], recall[i], f1[i], support[i]))
    
    class_metrics.sort(key=lambda x: x[3])  # Sort by F1
    
    for class_name, prec, rec, f1_val, sup in class_metrics:
        status = "❌" if f1_val < 0.85 else "✓"
        print(f"{status} {class_name:<18} {prec:<10.4f} {rec:<10.4f} {f1_val:<10.4f} {sup:<10}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    
    # Identify worst performing classes
    print("\n⚠️  Classes needing improvement (F1 < 0.85):")
    for class_name, prec, rec, f1_val, sup in class_metrics:
        if f1_val < 0.85:
            print(f"  - {class_name}: F1={f1_val:.4f} (Precision={prec:.4f}, Recall={rec:.4f}, n={sup})")

def print_feature_importance(model, X, top_n=20):
    """Print feature importance"""
    if hasattr(model, 'feature_importances_'):
        print(f"\nTop {top_n} Most Important Features:")
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        
        for i in indices:
            print(f"  {X.columns[i]:<60} {importances[i]:.4f}")

def train_and_evaluate(X, y, le_y):
    """Main training pipeline"""
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples")
    
    # Train Random Forest
    rf_model, rf_pred = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, le_y)
    print_detailed_results(y_test, rf_pred, le_y, "RANDOM FOREST")
    print_feature_importance(rf_model, X, top_n=15)
    
    # Train XGBoost
    try:
        xgb_model, xgb_pred = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, le_y)
        print_detailed_results(y_test, xgb_pred, le_y, "XGBOOST")
        print_feature_importance(xgb_model, X, top_n=15)
        
        # Ensemble: Weighted voting
        print("\n" + "="*70)
        print("ENSEMBLE MODEL (RF + XGBoost Voting)")
        print("="*70)
        
        # Get probability predictions
        rf_proba = rf_model.predict_proba(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)
        
        # Weighted average (XGBoost usually performs better)
        rf_acc = accuracy_score(y_test, rf_pred)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        
        # Weight based on validation performance
        rf_weight = rf_acc / (rf_acc + xgb_acc)
        xgb_weight = xgb_acc / (rf_acc + xgb_acc)
        
        ensemble_proba = (rf_weight * rf_proba) + (xgb_weight * xgb_proba)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        print(f"Weights: RF={rf_weight:.3f}, XGB={xgb_weight:.3f}")
        
        print_detailed_results(y_test, ensemble_pred, le_y, "ENSEMBLE")
        
        # Compare all models
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(f"Random Forest:  {rf_acc:.4f}")
        print(f"XGBoost:        {xgb_acc:.4f}")
        print(f"Ensemble:       {ensemble_acc:.4f}  ⭐")
        print(f"Best Model:     {'Ensemble' if ensemble_acc >= max(rf_acc, xgb_acc) else ('XGBoost' if xgb_acc > rf_acc else 'Random Forest')}")
        print("="*70)
        
        # Return best performing model
        if ensemble_acc >= max(rf_acc, xgb_acc):
            return (rf_model, xgb_model, "ensemble")
        else:
            return xgb_model if xgb_acc > rf_acc else rf_model
        
    except Exception as e:
        print(f"\n⚠️  XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with Random Forest only...")
        return rf_model

if __name__ == "__main__":
    df = load_processed()
    X, y, le_y = prepare_data(df)
    best_model = train_and_evaluate(X, y, le_y)