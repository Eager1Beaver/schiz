import os
import torch
import glob
import numpy as np
import pandas as pd
from logger import logger
import tqdm
from torch.utils.data import DataLoader
from utils.data_loader import MRIDataset
from models.cnn import FeatureExtractor


# Function to get image paths
def get_image_paths(folder_path):
    """Gets all .pt file paths from the specified folder."""
    image_paths = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.endswith('.pt')
    ]
    return image_paths


# Collate function for DataLoader (handles empty batches)
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels)


# Function to extract features using a pretrained model
def extract_features(loader, model, device):
    """Extracts features from MRI images using the specified model."""
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Extracting features"):
            if batch is None:
                continue  # Skip empty batches

            inputs, targets = batch
            inputs = inputs.to(device)
            outputs = model(inputs)

            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


# Main feature extraction pipeline
def feature_extraction_pipeline(raw_nii_dir: str = "data/raw_nii",
                                train_set_dir: str = "train_set",
                                test_set_dir: str = "test_set",
                                preprocessed_set_dir: str = "preprocessed",
                                labels_dir: str = "data/clinical_data.csv",
                                extracted_features_dir: str = "data/extracted_features",
                                target_shape: tuple = (224, 224),
                                batch_size: int = 32,
                                feature_extractor_model: str = 'resnet18',
                                base_model_weights = None,
                                input_channels: int = 1) -> None:
    os.makedirs(extracted_features_dir, exist_ok=True)

    # Search for all .pt files in all subdirectories under train_set_dir
    # This approach ensures that any new folders inside train_set_dir will automatically be included
    all_train_set_paths = glob.glob(os.path.join(raw_nii_dir, train_set_dir, '**', '*.pt'), recursive=True)

    test_set_dir = os.path.join(raw_nii_dir, test_set_dir, preprocessed_set_dir)
    all_test_set_paths = glob.glob(os.path.join(raw_nii_dir, test_set_dir, preprocessed_set_dir, '*.pt'), recursive=True)
    
    logger.info("Loading datasets...")
    train_dataset = MRIDataset(all_train_set_paths, pd.read_csv(labels_dir), target_shape=target_shape)
    test_dataset = MRIDataset(all_test_set_paths, pd.read_csv(labels_dir), target_shape=target_shape)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor(base_model_name=feature_extractor_model,
                             weights=base_model_weights,
                             input_channels=input_channels).to(device)

    # Extract features
    logger.info("Extracting training features...")
    train_features, train_labels = extract_features(train_loader, model,
                                                    device)
    np.save(os.path.join(extracted_features_dir, "train_features.npy"),
            train_features)
    np.save(os.path.join(extracted_features_dir, "train_labels.npy"),
            train_labels)

    logger.info("Extracting test features...")
    test_features, test_labels = extract_features(test_loader, model, device)
    np.save(os.path.join(extracted_features_dir, "test_features.npy"),
            test_features)
    np.save(os.path.join(extracted_features_dir, "test_labels.npy"),
            test_labels)

    logger.info("Feature extraction completed.")
