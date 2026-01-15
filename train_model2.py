import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

INPUT_PATH = "processed_dataset.csv"

def load_processed():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Processed dataset not found at {INPUT_PATH}. Run feature_engineering2.py first.")
    
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
        print(f"  {le_y.classes_[idx]}: {count}")
    
    return X, y_encoded, le_y

def apply_smote(X_train, y_train, strategy='auto'):
    """Apply SMOTE to balance minority classes"""
    print("\nApplying SMOTE to balance classes...")
    print(f"Before SMOTE: {Counter(y_train)}")
    
    # SMOTE with careful strategy
    smote = SMOTE(
        sampling_strategy=strategy,
        random_state=42,
        k_neighbors=5
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {Counter(y_resampled)}")
    
    return X_resampled, y_resampled

def train_validate_test(X, y, le_y, use_smote=True):
    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Apply SMOTE to training data only
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, strategy='not majority')
    
    # Optimized Random Forest with better hyperparameters
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=1400,              # More trees for better performance
        max_depth=None,                  # Limit depth to prevent overfitting
        min_samples_split=3,          # Require more samples to split
        min_samples_leaf=1,            # Require more samples in leaf
        max_features='log2',           # Use sqrt of features at each split
        class_weight='balanced',       # Balance classes
        bootstrap=True,
        oob_score=True,                # Out-of-bag score
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    
    if rf.oob_score_:
        print(f"Out-of-Bag Score: {rf.oob_score_:.4f}")
    
    # Validation
    val_pred = rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    print(f"\n{'='*60}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")
    
    # Test
    test_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"{'='*60}\n")
    
    # Detailed classification report
    print("Classification Report (Test Set):")
    print(classification_report(
        y_test, test_pred, 
        target_names=le_y.classes_,
        digits=4
    ))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    
    # Per-class analysis
    print("\nPer-Class Analysis:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, test_pred, labels=range(len(le_y.classes_))
    )
    
    for i, class_name in enumerate(le_y.classes_):
        print(f"{class_name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # Feature importance
    print("\nTop 15 Most Important Features:")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-15:][::-1]
    feature_names = X.columns
    
    for i in indices:
        print(f"  {feature_names[i]:<60} {importances[i]:.4f}")
    
    # Cross-validation for robustness check
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(
        rf, X_train, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=1
    )
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return rf

if __name__ == "__main__":
    df = load_processed()
    X, y, le_y = prepare_data(df)
    
    # Train with SMOTE
    print("\n" + "="*60)
    print("TRAINING WITH SMOTE")
    print("="*60)
    model = train_validate_test(X, y, le_y, use_smote=True)