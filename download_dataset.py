# download_dataset.py
# This file downloads the dataset from a public GitHub source (full merged CSV)
# Run this first to get 'full-dataset-CCCS-CIC-AndMal-2020.csv' in your folder

import requests
import os

# URL to the raw CSV file (from a public GitHub repo)
DATASET_URL = "https://raw.githubusercontent.com/panicoro/ML-CCCS-CIC-AndMal-2020/master/full-dataset-CCCS-CIC-AndMal-2020.csv"
SAVE_PATH = "full-dataset-CCCS-CIC-AndMal-2020.csv"

def download_dataset():
    if os.path.exists(SAVE_PATH):
        print(f"File already exists: {SAVE_PATH}")
        return
    
    print(f"Downloading dataset from {DATASET_URL}...")
    response = requests.get(DATASET_URL)
    
    if response.status_code == 200:
        with open(SAVE_PATH, 'wb') as f:
            f.write(response.content)
        print(f"Dataset downloaded and saved to {SAVE_PATH}")
    else:
        print(f"Error downloading: Status code {response.status_code}")
        print("Please download manually from https://www.kaggle.com/datasets/dhoogla/cccscicandmal2020 or official UNB site.")

if __name__ == "__main__":
    download_dataset()