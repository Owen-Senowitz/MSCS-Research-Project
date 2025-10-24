import os
from dotenv import load_dotenv

# Load Kaggle credentials from .env
load_dotenv("../../.env")

username = os.getenv("KAGGLE_USERNAME")
key = os.getenv("KAGGLE_KEY")

from kaggle.api.kaggle_api_extended import KaggleApi

print("Using Kaggle account:", username)

# Set Kaggle credentials as environment variables for KaggleApi
os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = key

# Directory where you want the dataset
dataset_dir = "../../data"

# Check if dataset already exists
if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
    print(f"Dataset already exists in {dataset_dir}, skipping download.")
else:
    print("Dataset not found, downloading now...")

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        "awsaf49/cbis-ddsm-breast-cancer-image-dataset",
        path=dataset_dir,
        unzip=True
    )

    print(f"Dataset downloaded to {dataset_dir}")
