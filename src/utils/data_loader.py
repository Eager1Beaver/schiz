import re
import torch
import logging
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Optional


# Class to return each slice and the label
class MRIDataset(Dataset):
    """
    A PyTorch Dataset for loading MRI slices and their corresponding labels.

    This dataset loads MRI scans from tensor files, extracts 2D slices, and resizes them.
    Labels are matched using a provided CSV file that contains participant IDs.

    Attributes:
        image_paths (List[Path]): List of file paths to MRI tensor files.
        labels_df (pd.DataFrame): DataFrame containing participant IDs and corresponding labels.
        target_shape (Tuple[int, int]): Desired shape (height, width) for each 2D slice.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels_file: pd.DataFrame,
        target_shape: Tuple[int, int] = (224, 224)
    ) -> None:
        """
            Initializes the dataset.

            Parameters:
                image_paths (List[str]): List of file paths to MRI tensors.
                labels_file (pd.DataFrame): DataFrame containing participant IDs and labels.
                target_shape (Tuple[int, int]): Target height and width for 2D slices.
                
            Returns:
                None
            """
        self.image_paths = [Path(p)
                            for p in image_paths]  # Convert to Path objects
        self.labels_df = labels_file
        self.target_shape = target_shape

    def __len__(self) -> int:
        """
            Returns the total number of slices across all MRI files.

            Returns:
                int: Total number of 2D slices.
            """
        total_slices = 0
        for path in self.image_paths:
            mri_tensor = torch.load(path, weights_only=True)
            total_slices += mri_tensor.shape[
                0]  # Assume the first dimension is slices
        return total_slices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
            Retrieves a single resized 2D MRI slice and its corresponding label.

            Parameters:
                idx (int): Index of the slice to retrieve.

            Returns:
                Tuple[torch.Tensor, int]: The processed 2D MRI slice and its label.
            """
        accumulated_slices = 0
        file_path: Optional[Path] = None
        slice_index: Optional[int] = None

        # Find the MRI file corresponding to the requested index
        for path in self.image_paths:
            mri_tensor = torch.load(path, weights_only=True)
            num_slices = mri_tensor.shape[
                0]  # Number of slices in the current file
            if idx < accumulated_slices + num_slices:
                file_path = path
                slice_index = idx - accumulated_slices
                break
            accumulated_slices += num_slices

        if file_path is None or slice_index is None:
            raise IndexError(f"Index {idx} is out of range.")

        # Extract subject number and label
        subject_number = self._extract_subject_number(file_path.name)
        label_row = self.labels_df[self.labels_df["participant_id"] ==
                                   subject_number]

        if label_row.empty:
            logging.warning(f"No label found for subject: {subject_number}")
            raise ValueError(f"Missing label for subject: {subject_number}")

        label = int(label_row["dx_encoded"].values[0])

        # Load the MRI tensor and extract the specific slice
        try:
            mri_tensor = torch.load(file_path, weights_only=True)
            slice_2d = mri_tensor[slice_index, :, :]  # Extract the 2D slice
            slice_2d = slice_2d.unsqueeze(
                0)  # Add channel dimension for PyTorch

            # Resize the slice to the target shape
            slice_2d = F.interpolate(slice_2d.unsqueeze(0),
                                     size=self.target_shape,
                                     mode="bilinear",
                                     align_corners=False)
            slice_2d = slice_2d.squeeze(0)  # Remove the batch dimension
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            raise RuntimeError(f"Error processing file {file_path}: {e}")

        return slice_2d, label

    @staticmethod
    def _extract_subject_number(file_name: str) -> Optional[str]:
        """
        Extracts the subject number from the MRI file name.

        Parameters:
            file_name (str): The name of the file (e.g., "sub-A123.nii.gz").

        Returns:
            Optional[str]: The extracted subject number if found, otherwise None.
        """
        match = re.search(r'sub-(A\d+)', file_name)
        return match.group(1) if match else None
