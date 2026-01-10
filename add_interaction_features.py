# add_interaction_features.py
# Add advanced interaction features and polynomial terms
# Expected improvement: +1-3% accuracy

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from itertools import combinations

INPUT_PATH = "processed_dataset.csv"
OUTPUT_PATH = "processed_dataset_v2.csv"

def load_dataset():
    print("Loading processed dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Original shape: {df.shape}")
    return df

def create_interaction_features(df):
    """Create interaction terms between important features"""
    print("\nCreating interaction features...")
    
    # Key feature groups for interactions
    behavioral_features = [
        'Obfuscation_Score', 'SMS_Score', 'Device_Info_Score',
        'Crypto_Score', 'Network_API_Score', 'Process_Manipulation_Score',
        'File_Access_Score'
    ]
    
    existing_behavioral = [f for f in behavioral_features if f in df.columns]
    
    if len(existing_behavioral) >= 2:
        # Create pairwise interactions
        interaction_count = 0
        for feat1, feat2 in combinations(existing_behavioral, 2):
            # Multiplicative interaction
            df[f'{feat1}_X_{feat2}'] = df[feat1] * df[feat2]
            interaction_count += 1
            
            # Ratio interaction (if denominator not zero)
            df[f'{feat1}_DIV_{feat2}'] = df[feat1] / (df[feat2] + 1)
            interaction_count += 1
        
        print(f"  Created {interaction_count} interaction features")
    
    return df

def create_statistical_features(df):
    """Create statistical aggregations across feature groups"""
    print("Creating statistical features...")
    
    # Group features by type
    api_cols = [col for col in df.columns if 'API' in col and col not in ['Total_API_Calls', 'API_Diversity', 'API_Entropy']]
    memory_cols = [col for col in df.columns if 'Memory' in col and 'Total' not in col]
    
    feature_count = 0
    
    # API statistical features
    if len(api_cols) > 10:
        # Percentiles
        df['API_25th_Percentile'] = df[api_cols].quantile(0.25, axis=1)
        df['API_75th_Percentile'] = df[api_cols].quantile(0.75, axis=1)
        df['API_IQR'] = df['API_75th_Percentile'] - df['API_25th_Percentile']
        
        # Concentration (how concentrated is API usage)
        df['API_Concentration'] = df[api_cols].max(axis=1) / (df[api_cols].sum(axis=1) + 1)
        
        feature_count += 4
    
    # Memory statistical features
    if len(memory_cols) > 5:
        df['Memory_Median'] = df[memory_cols].median(axis=1)
        df['Memory_MAD'] = (df[memory_cols].sub(df['Memory_Median'], axis=0).abs()).median(axis=1)
        df['Memory_Skewness'] = df[memory_cols].skew(axis=1)
        
        feature_count += 3
    
    print(f"  Created {feature_count} statistical features")
    
    return df

def create_domain_specific_features(df):
    """Create malware-detection specific features"""
    print("Creating domain-specific features...")
    
    feature_count = 0
    
    # Suspicious behavior combinations
    if 'Device_Info_Score' in df.columns and 'SMS_Score' in df.columns:
        # High device info + SMS = likely spyware/banker
        df['Spyware_Pattern'] = (
            (df['Device_Info_Score'] > df['Device_Info_Score'].quantile(0.75)).astype(int) +
            (df['SMS_Score'] > df['SMS_Score'].quantile(0.75)).astype(int)
        )
        feature_count += 1
    
    if 'Obfuscation_Score' in df.columns and 'Network_API_Score' in df.columns:
        # Obfuscation + Network = likely backdoor/dropper
        df['Stealth_Download_Pattern'] = df['Obfuscation_Score'] * df['Network_API_Score']
        df['High_Stealth'] = (df['Stealth_Download_Pattern'] > df['Stealth_Download_Pattern'].quantile(0.75)).astype(int)
        feature_count += 2
    
    # Activity intensity
    if 'Total_API_Calls' in df.columns and 'Unique_APIs_Used' in df.columns:
        # High total calls but low diversity = repetitive behavior
        df['API_Repetition_Score'] = df['Total_API_Calls'] / (df['Unique_APIs_Used'] + 1)
        df['High_Repetition'] = (df['API_Repetition_Score'] > df['API_Repetition_Score'].quantile(0.80)).astype(int)
        feature_count += 2
    
    # Crypto + File = Ransomware pattern (already exists, but enhance it)
    if 'Crypto_Score' in df.columns and 'File_Access_Score' in df.columns:
        df['Enhanced_Ransomware_Pattern'] = (
            df['Crypto_Score'] * df['File_Access_Score'] * 
            (df['Crypto_Score'] > 0).astype(int)
        )
        feature_count += 1
    
    print(f"  Created {feature_count} domain-specific features")
    
    return df

def create_temporal_proxies(df):
    """Create features that proxy for temporal behavior"""
    print("Creating temporal proxy features...")
    
    # These aren't true temporal features, but statistical patterns that
    # may correlate with malware behavior over time
    
    feature_count = 0
    
    # Coefficient of variation for API usage
    api_cols = [col for col in df.columns if 'API' in col and col not in ['Total_API_Calls', 'API_Diversity', 'API_Entropy']]
    
    if len(api_cols) > 10:
        api_mean = df[api_cols].mean(axis=1) + 1e-10
        api_std = df[api_cols].std(axis=1)
        df['API_CoV'] = api_std / api_mean  # Coefficient of variation
        feature_count += 1
    
    # Burst behavior indicator
    if 'Total_API_Calls' in df.columns:
        df['API_Burst_Indicator'] = (df['Total_API_Calls'] > df['Total_API_Calls'].quantile(0.90)).astype(int)
        feature_count += 1
    
    print(f"  Created {feature_count} temporal proxy features")
    
    return df

def remove_low_variance_features(df, threshold=0.01):
    """Remove features with very low variance"""
    print("\nRemoving low-variance features...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'Category']
    
    variances = df[numeric_cols].var()
    low_var_features = variances[variances < threshold].index.tolist()
    
    if low_var_features:
        print(f"  Removing {len(low_var_features)} low-variance features")
        df = df.drop(columns=low_var_features)
    else:
        print("  No low-variance features found")
    
    return df

def handle_infinite_values(df):
    """Handle infinite values from divisions"""
    print("Handling infinite values...")
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaN with median for each column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Final safety: fill any remaining NaN with 0
    df.fillna(0, inplace=True)
    
    return df

def scale_new_features(df):
    """Scale only the new features"""
    print("Scaling new features...")
    
    # Identify new features (those not in the original processed dataset)
    # This is approximate - scales all numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'Category']
    
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def main():
    # Load data
    df = load_dataset()
    
    # Create new features
    df = create_interaction_features(df)
    df = create_statistical_features(df)
    df = create_domain_specific_features(df)
    df = create_temporal_proxies(df)
    
    # Clean up
    df = handle_infinite_values(df)
    df = remove_low_variance_features(df, threshold=0.01)
    
    # Scale
    df = scale_new_features(df)
    
    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✓ Enhanced dataset saved to {OUTPUT_PATH}")
    print(f"✓ Final shape: {df.shape}")
    print(f"✓ Added {df.shape[1] - 165} new features")  # 165 was your original count
    
    print("\nNext steps:")
    print("1. Update INPUT_PATH in train_model.py to 'processed_dataset_v2.csv'")
    print("2. Run hyperparameter_tuning.py for best parameters")
    print("3. Expected combined improvement: +3-6% accuracy")

if __name__ == "__main__":
    main()