# feature_engineering_optimal.py
# KEEP ALL ORIGINAL FEATURES + ADD BEHAVIORAL PATTERNS

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
import os

INPUT_PATH = "full-dataset-CCCS-CIC-AndMal-2020.csv"
OUTPUT_PATH = "processed_dataset.csv"

# LEAKAGE_FEATURES = ['sha256', 'pkg_name', 'timestamp']  # Keep Family for 0.93 (DATA LEAKAGE!)
LEAKAGE_FEATURES = ['Family', 'sha256', 'pkg_name', 'timestamp']

def load_dataset():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Dataset not found at {INPUT_PATH}.")
    
    print(f"Loading dataset from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Original shape: {df.shape}")
    print(f"Sample columns: {df.columns[:20].tolist()}")
    return df

def remove_leakage_features(df):
    """Remove ONLY leakage features, keep everything else"""
    leaked = [col for col in LEAKAGE_FEATURES if col in df.columns]
    if leaked:
        print(f"⚠️  Removing leakage features: {leaked}")
        df = df.drop(columns=leaked)
    return df

def add_behavioral_features(df):
    """ADD new features without removing original ones"""
    print("\n🔧 Adding behavioral pattern features...")
    
    # Identify feature groups
    api_cols = [col for col in df.columns if 'API' in col]
    memory_cols = [col for col in df.columns if 'Memory' in col]
    network_cols = [col for col in df.columns if 'Network' in col]
    
    print(f"Found {len(api_cols)} API features")
    print(f"Found {len(memory_cols)} Memory features")
    print(f"Found {len(network_cols)} Network features")
    
    # === API BEHAVIORAL PATTERNS ===
    if len(api_cols) > 0:
        # Total activity
        df['Total_API_Calls'] = df[api_cols].sum(axis=1)
        df['Unique_APIs_Used'] = (df[api_cols] > 0).sum(axis=1)
        df['API_Diversity'] = df['Unique_APIs_Used'] / (len(api_cols) + 1)
        
        # API entropy
        api_sum = df[api_cols].sum(axis=1) + 1e-10
        api_probs = df[api_cols].div(api_sum, axis=0)
        df['API_Entropy'] = -(api_probs * np.log2(api_probs + 1e-10)).sum(axis=1)
        
        # Malware-specific API patterns
        # Obfuscation/Dynamic Loading (Zero-Day, Trojan Dropper)
        obfuscation_keywords = ['DexClassLoader', 'DexFile', 'Reflection', 'ClassLoader', 'Runtime']
        obfuscation_cols = [col for col in api_cols if any(k in col for k in obfuscation_keywords)]
        if obfuscation_cols:
            df['Obfuscation_Score'] = df[obfuscation_cols].sum(axis=1)
            df['Uses_Obfuscation'] = (df['Obfuscation_Score'] > 0).astype(int)
        
        # SMS/Banking (Trojan_Banker, Trojan_SMS)
        sms_keywords = ['SMS', 'sendTextMessage', 'SmsManager', 'getLine1Number']
        sms_cols = [col for col in api_cols if any(k in col for k in sms_keywords)]
        if sms_cols:
            df['SMS_Score'] = df[sms_cols].sum(axis=1)
            df['High_SMS_Activity'] = (df['SMS_Score'] > 5).astype(int)
        
        # Device Info Collection (Spyware, Adware)
        device_keywords = ['TelephonyManager', 'getDeviceId', 'getSubscriberId', 'getIMEI', 'getSimSerialNumber']
        device_cols = [col for col in api_cols if any(k in col for k in device_keywords)]
        if device_cols:
            df['Device_Info_Score'] = df[device_cols].sum(axis=1)
            df['Collects_Device_Info'] = (df['Device_Info_Score'] > 3).astype(int)
        
        # Crypto/Hashing (Ransomware, Backdoor)
        crypto_keywords = ['Crypto', 'Cipher', 'Hash', 'MessageDigest', 'encrypt']
        crypto_cols = [col for col in api_cols if any(k in col for k in crypto_keywords)]
        if crypto_cols:
            df['Crypto_Score'] = df[crypto_cols].sum(axis=1)
            df['Uses_Crypto'] = (df['Crypto_Score'] > 0).astype(int)
        
        # Network/Download (Backdoor, Trojan Dropper)
        network_keywords = ['HttpURLConnection', 'DownloadManager', 'URLConnection', 'Socket']
        network_api_cols = [col for col in api_cols if any(k in col for k in network_keywords)]
        if network_api_cols:
            df['Network_API_Score'] = df[network_api_cols].sum(axis=1)
            df['High_Network_Activity'] = (df['Network_API_Score'] > 5).astype(int)
        
        # Process/Service Manipulation (Backdoor, Scareware)
        process_keywords = ['killBackgroundProcesses', 'Runtime_exec', 'ProcessBuilder', 'startService', 'stopService']
        process_cols = [col for col in api_cols if any(k in col for k in process_keywords)]
        if process_cols:
            df['Process_Manipulation_Score'] = df[process_cols].sum(axis=1)
        
        # File System Access (FileInfector)
        file_keywords = ['FileOutputStream', 'FileInputStream', 'File_delete', 'File_write']
        file_cols = [col for col in api_cols if any(k in col for k in file_keywords)]
        if file_cols:
            df['File_Access_Score'] = df[file_cols].sum(axis=1)
    
    # === MEMORY PATTERNS ===
    if len(memory_cols) >= 2:
        df['Total_Memory'] = df[memory_cols].sum(axis=1)
        df['Memory_Variance'] = df[memory_cols].var(axis=1)
        df['Memory_Max_Min_Ratio'] = df[memory_cols].max(axis=1) / (df[memory_cols].min(axis=1) + 1)
    
    # === NETWORK PATTERNS ===
    if len(network_cols) >= 2:
        df['Total_Network_Traffic'] = df[network_cols].sum(axis=1)
        
        sent_cols = [col for col in network_cols if 'Transmitted' in col or 'Sent' in col]
        recv_cols = [col for col in network_cols if 'Received' in col or 'Recv' in col]
        if sent_cols and recv_cols:
            df['Network_Send_Recv_Ratio'] = df[sent_cols].sum(axis=1) / (df[recv_cols].sum(axis=1) + 1)
    
    # === INTERACTION PATTERNS ===
    # Combine behaviors that are suspicious together
    if 'Device_Info_Score' in df.columns and 'Network_API_Score' in df.columns:
        df['Data_Exfiltration_Risk'] = df['Device_Info_Score'] * df['Network_API_Score']
    
    if 'Crypto_Score' in df.columns and 'File_Access_Score' in df.columns:
        df['Ransomware_Pattern'] = df['Crypto_Score'] * df['File_Access_Score']
    
    new_features = len([c for c in df.columns if c not in api_cols + memory_cols + network_cols + ['Category']])
    print(f"✓ Added {new_features} new behavioral features")
    print(f"Total features now: {df.shape[1]}")
    
    return df

def handle_missing_and_outliers(df):
    """Handle missing values and outliers"""
    print("\nHandling missing values...")
    
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
    """Scale numerical features"""
    print("Scaling features with RobustScaler...")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != 'Category']
    
    scaler = RobustScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def preprocess_and_engineer(df):
    """Main preprocessing pipeline - KEEPS ALL ORIGINAL FEATURES"""
    print("\n" + "="*70)
    print("OPTIMAL FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # 1. Remove ONLY leakage features
    df = remove_leakage_features(df)
    print(f"After removing leakage: {df.shape}")
    
    # 2. Encode categorical first
    df = encode_categorical(df)
    
    # 3. Handle missing values
    df = handle_missing_and_outliers(df)
    
    # 4. ADD behavioral features (don't replace!)
    df = add_behavioral_features(df)
    
    # 5. Handle any new missing values
    df = handle_missing_and_outliers(df)
    
    # 6. Scale ALL features
    df = scale_features(df)
    
    print(f"\n✓ Final processed shape: {df.shape}")
    print("="*70)
    
    return df

def save_processed(df):
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Processed dataset saved to {OUTPUT_PATH}")
    
    if 'Category' in df.columns:
        print("\nClass distribution:")
        print(df['Category'].value_counts().sort_index())

if __name__ == "__main__":
    df = load_dataset()
    processed_df = preprocess_and_engineer(df)
    save_processed(processed_df)
    print(f"\n🎯 Dataset ready for training with {processed_df.shape[1]-1} features!")