import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset
from src.utils.preprocess import load_nii, preprocess_image

class MRIDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Custom Data Loader to load MRI data and labels.

        Args:
            data_path (str): Path to the top-level "data" folder.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data_path = data_path
        self.transform = transform
        self.file_paths, self.labels = self._load_data_and_labels()

    def _load_data_and_labels(self):
        """
        Load all .nii.gz file paths and their corresponding labels based on CSV files.
        
        Returns:
            file_paths (list): List of paths to .nii.gz files.
            labels (list): List of labels (0 or 1) corresponding to each file (each subject).
        """
        file_paths = []
        labels = []

        # Iterate over COBRE and MCICShare folders
        for dataset_name in ["COBRE", "MCICShare"]:
            dataset_full_name = "schizconnect_" + dataset_name + "_images" + "_22613"
            dataset_path = os.path.join(self.data_path, dataset_full_name, dataset_name)
            csv_path = os.path.join(dataset_path, "participants.csv")

        # Load participant mapping from CSV
        participants_df = pd.read_csv(csv_path)
        participants_df.set_index(participants_df["participant_id"].str.replace("sub-", "", regex=False), inplace=True)

        # Find all .nii.gz files in the anat subfolder
        anat_files = glob.glob(os.path.join(dataset_path, "sub-*", "ses-*", "anat", "*.nii.gz"))

        for file_path in anat_files:
            # Extract subject ID (e.g., "A00000300" from "sub-A00000300")
            subject_id = os.path.basename(file_path).split("_")[0].replace("sub-", "")

            # Retrieve the label (0 or 1) from the CSV
            if subject_id in participants_df.index:
                label = participants_df.loc[subject_id, "dx_encoded"]
                file_paths.append(file_path)
                labels.append(label)
            else: # TODO: collect subjects with wrong labels or mismatches?
                print(f"Warning: No label found for {subject_id} in {csv_path}")

        return file_paths, labels    

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Get an image and its corresponding label.

        Args:
            idx (int): Index of the sample.

        Returns:
            image (torch.Tensor): Preprocessed MRI image.
            label (int): Corresponding label (0 or 1).
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load and preprocess the image
        image = load_nii(file_path)
        image = preprocess_image(image)

        # Apply the transformations if provided
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float()  # Convert back to tensor conversion if no transform
    
        return image, label