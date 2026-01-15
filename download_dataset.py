import requests
import os

# URL to the raw CSV file
DATASET_URL = "https://raw.githubusercontent.com/panicoro/ML-CCCS-CIC-AndMal-2020/master/full-dataset-CCCS-CIC-AndMal-2020.csv"
SAVE_PATH = "full-dataset-CCCS-CIC-AndMal-2020.csv"

def download_dataset():
    if os.path.exists(SAVE_PATH):
        print(f"File already exists: {SAVE_PATH}")
        return
    
    print(f"Downloading dataset from {DATASET_URL}...")
    
    try:
        # 1. Use stream=True to keep connection open without downloading immediately
        response = requests.get(DATASET_URL, stream=True)
        
        if response.status_code == 200:
            with open(SAVE_PATH, 'wb') as f:
                # 2. Download in chunks (e.g., 8KB at a time)
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Dataset downloaded and saved to {SAVE_PATH}")
        else:
            print(f"Error downloading: Status code {response.status_code}")
            
    except Exception as e:
        print(f"Download failed: {e}")
        print("Try downloading manually from the URL in the code.")

if __name__ == "__main__":
    download_dataset()