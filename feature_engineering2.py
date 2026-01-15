import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import os

INPUT_PATH = "full-dataset-CCCS-CIC-AndMal-2020.csv"
OUTPUT_PATH = "processed_dataset.csv"

# CRITICAL: Features that may cause data leakage
LEAKAGE_FEATURES = ['Family', 'sha256', 'pkg_name', 'timestamp']

def load_dataset():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Dataset not found at {INPUT_PATH}. Run download_dataset.py first.")
    
    print(f"Loading dataset from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def remove_leakage_features(df):
    """Remove features that may leak target information"""
    leaked = [col for col in LEAKAGE_FEATURES if col in df.columns]
    if leaked:
        print(f"⚠️  REMOVING POTENTIAL LEAKAGE FEATURES: {leaked}")
        df = df.drop(columns=leaked)
    return df

def advanced_feature_engineering(df):
    """Create sophisticated derived features"""
    print("Creating advanced features...")
    
    # Memory-related features
    memory_cols = [col for col in df.columns if 'Memory' in col or 'memory' in col]
    if len(memory_cols) >= 2:
        # Total memory usage
        df['Total_Memory'] = df[memory_cols].sum(axis=1)
        # Memory variance (complexity indicator)
        df['Memory_Variance'] = df[memory_cols].var(axis=1)
        # Memory max/min ratio
        df['Memory_Range_Ratio'] = df[memory_cols].max(axis=1) / (df[memory_cols].min(axis=1) + 1)
    
    # Network-related features
    network_cols = [col for col in df.columns if 'Network' in col or 'network' in col]
    if len(network_cols) >= 2:
        df['Total_Network_Traffic'] = df[network_cols].sum(axis=1)
        df['Network_Variance'] = df[network_cols].var(axis=1)
        
        # Sent/Received ratio if available
        sent_cols = [col for col in network_cols if 'Transmitted' in col or 'Sent' in col]
        recv_cols = [col for col in network_cols if 'Received' in col or 'Recv' in col]
        if sent_cols and recv_cols:
            df['Network_Send_Recv_Ratio'] = df[sent_cols].sum(axis=1) / (df[recv_cols].sum(axis=1) + 1)
    
    # API call features
    api_cols = [col for col in df.columns if 'API' in col or 'api' in col]
    if len(api_cols) > 0:
        # Total API calls
        df['Total_API_Calls'] = df[api_cols].sum(axis=1)
        # Number of unique APIs used
        df['Unique_APIs_Used'] = (df[api_cols] > 0).sum(axis=1)
        # API diversity (entropy-like measure)
        df['API_Diversity'] = df['Unique_APIs_Used'] / (len(api_cols) + 1)
        
        # Suspicious API patterns
        crypto_apis = [col for col in api_cols if 'Crypto' in col or 'crypto' in col]
        device_apis = [col for col in api_cols if 'Device' in col or 'device' in col]
        if crypto_apis:
            df['Crypto_API_Usage'] = df[crypto_apis].sum(axis=1)
        if device_apis:
            df['Device_API_Usage'] = df[device_apis].sum(axis=1)
    
    # Statistical features across all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'Category']
    
    if len(numeric_cols) > 5:
        # Aggregated statistics
        df['Feature_Mean'] = df[numeric_cols].mean(axis=1)
        df['Feature_Std'] = df[numeric_cols].std(axis=1)
        df['Feature_Skew'] = df[numeric_cols].skew(axis=1)
        df['Feature_Kurtosis'] = df[numeric_cols].kurtosis(axis=1)
        df['Non_Zero_Features'] = (df[numeric_cols] != 0).sum(axis=1)
    
    print(f"Created {df.shape[1] - len(numeric_cols)} new features")
    return df

def handle_missing_and_outliers(df):
    """Advanced missing value and outlier handling"""
    print("Handling missing values and outliers...")
    
    # Fill missing values with median (more robust than mean)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def encode_categorical(df):
    """Encode categorical features"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Category']
    
    if len(categorical_cols) > 0:
        print(f"Encoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df

def scale_features(df):
    """Use RobustScaler (better for outliers than StandardScaler)"""
    print("Scaling features with RobustScaler...")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != 'Category']
    
    scaler = RobustScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def feature_selection(df, n_features=100):
    """Select top features using mutual information"""
    print(f"Selecting top {n_features} features using mutual information...")
    
    if 'Category' not in df.columns:
        print("Warning: Category column not found, skipping feature selection")
        return df
    
    X = df.drop(columns=['Category'])
    y = df['Category']
    
    # Encode target for feature selection
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Select features
    selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
    selector.fit(X, y_encoded)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    selected_features.append('Category')
    
    print(f"Selected {len(selected_features)-1} features")
    return df[selected_features]

def preprocess_and_engineer(df):
    """Main preprocessing pipeline"""
    # 1. Remove data leakage features
    df = remove_leakage_features(df)
    
    # 2. Handle missing values early
    df = handle_missing_and_outliers(df)
    
    # 3. Encode categorical features
    df = encode_categorical(df)
    
    # 4. Advanced feature engineering
    df = advanced_feature_engineering(df)
    
    # 5. Handle any new missing values from feature engineering
    df = handle_missing_and_outliers(df)
    
    # 6. Feature selection (optional, comment out if you want all features)
    # df = feature_selection(df, n_features=120)
    
    # 7. Scale features
    df = scale_features(df)
    
    print(f"Final processed shape: {df.shape}")
    return df

def save_processed(df):
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Processed dataset saved to {OUTPUT_PATH}")
    
    # Print class distribution
    if 'Category' in df.columns:
        print("\nClass distribution:")
        print(df['Category'].value_counts().sort_index())

if __name__ == "__main__":
    df = load_dataset()
    processed_df = preprocess_and_engineer(df)
    save_processed(processed_df)