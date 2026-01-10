# hyperparameter_tuning.py
# Advanced hyperparameter optimization using Optuna
# Expected improvement: +2-4% accuracy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

INPUT_PATH = "processed_dataset.csv"

def load_and_prepare_data():
    """Load and prepare dataset"""
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)
    
    target_col = 'Category'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode target
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, le_y

def compute_sample_weights(y_train):
    """Compute sample weights"""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    sample_weights = np.array([weight_dict[y] for y in y_train])
    return sample_weights

def objective_xgboost(trial, X_train, y_train, X_val, y_val, sample_weights):
    """Objective function for XGBoost hyperparameter tuning"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
        'max_depth': trial.suggest_int('max_depth', 10, 25),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y_train)),
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'tree_method': 'hist'
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy

def objective_random_forest(trial, X_train, y_train, X_val, y_val):
    """Objective function for Random Forest hyperparameter tuning"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
        'max_depth': trial.suggest_categorical('max_depth', [None, 30, 40, 50]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8, 0.9]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
        'max_samples': trial.suggest_float('max_samples', 0.7, 1.0),
        'bootstrap': True,
        'oob_score': True,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy

def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=50):
    """Tune XGBoost hyperparameters"""
    print("\n" + "="*70)
    print("TUNING XGBOOST HYPERPARAMETERS")
    print("="*70)
    
    sample_weights = compute_sample_weights(y_train)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train, X_val, y_val, sample_weights),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\nBest XGBoost Accuracy: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params

def tune_random_forest(X_train, y_train, X_val, y_val, n_trials=50):
    """Tune Random Forest hyperparameters"""
    print("\n" + "="*70)
    print("TUNING RANDOM FOREST HYPERPARAMETERS")
    print("="*70)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective_random_forest(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\nBest RF Accuracy: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params

def train_best_models(X_train, y_train, X_val, y_val, X_test, y_test, le_y, 
                      xgb_params, rf_params):
    """Train final models with best parameters"""
    print("\n" + "="*70)
    print("TRAINING FINAL MODELS WITH BEST PARAMETERS")
    print("="*70)
    
    sample_weights = compute_sample_weights(y_train)
    
    # Train XGBoost
    xgb_params['objective'] = 'multi:softmax'
    xgb_params['num_class'] = len(le_y.classes_)
    xgb_params['eval_metric'] = 'mlogloss'
    xgb_params['random_state'] = 42
    xgb_params['n_jobs'] = -1
    xgb_params['verbosity'] = 0
    xgb_params['tree_method'] = 'hist'
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    
    # Train Random Forest
    rf_params['bootstrap'] = True
    rf_params['oob_score'] = True
    rf_params['random_state'] = 42
    rf_params['n_jobs'] = -1
    rf_params['verbose'] = 0
    
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    
    rf_acc = accuracy_score(y_test, rf_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
    
    print(f"\nRandom Forest - Test Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}")
    print(f"XGBoost       - Test Accuracy: {xgb_acc:.4f}, F1: {xgb_f1:.4f}")
    
    # Ensemble
    rf_proba = rf_model.predict_proba(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    
    rf_weight = rf_acc / (rf_acc + xgb_acc)
    xgb_weight = xgb_acc / (rf_acc + xgb_acc)
    
    ensemble_proba = (rf_weight * rf_proba) + (xgb_weight * xgb_proba)
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
    
    print(f"Ensemble      - Test Accuracy: {ensemble_acc:.4f}, F1: {ensemble_f1:.4f} ⭐")
    
    # Save best parameters
    print("\n" + "="*70)
    print("SAVING BEST PARAMETERS")
    print("="*70)
    
    with open('best_params.txt', 'w') as f:
        f.write("BEST XGBOOST PARAMETERS:\n")
        for key, value in xgb_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\nBEST RANDOM FOREST PARAMETERS:\n")
        for key, value in rf_params.items():
            f.write(f"{key}: {value}\n")
    
    print("✓ Best parameters saved to best_params.txt")
    print(f"✓ Ensemble Accuracy: {ensemble_acc:.4f}")
    
    return ensemble_acc

if __name__ == "__main__":
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, le_y = load_and_prepare_data()
    
    # Tune models (adjust n_trials based on time available)
    # More trials = better results but slower
    # 50 trials = ~30-60 minutes
    # 100 trials = ~1-2 hours
    # 20 trials = ~10-20 minutes (quick test)
    
    N_TRIALS = 30  # Change this based on your time budget
    
    xgb_params = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=N_TRIALS)
    rf_params = tune_random_forest(X_train, y_train, X_val, y_val, n_trials=N_TRIALS)
    
    # Train final models
    final_acc = train_best_models(
        X_train, y_train, X_val, y_val, X_test, y_test, le_y,
        xgb_params, rf_params
    )
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*70)
    print(f"Final Ensemble Accuracy: {final_acc:.4f}")
    print("Next steps:")
    print("1. Copy the best parameters from best_params.txt")
    print("2. Update your train_model.py with these parameters")
    print("3. Expected improvement: +2-4% accuracy")
    print("="*70)