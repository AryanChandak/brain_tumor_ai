import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate using kaggle.json
api = KaggleApi()
api.authenticate()

# Dataset name and path
dataset = 'sartajbhuvaji/brain-tumor-classification-mri'
output_path = 'data/mri_images'

# Create folder if not exists
os.makedirs(output_path, exist_ok=True)

# Download and extract
api.dataset_download_files(dataset, path=output_path, unzip=True)

print(f"✅ Dataset downloaded and extracted to: {output_path}")
