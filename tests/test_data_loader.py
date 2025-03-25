import unittest
import torch
import pandas as pd
import tempfile
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.data_loader import MRIDataset  # Ensure this is correctly imported

class TestMRIDataset(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for fake MRI tensor files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mri_files = []

        # Generate synthetic MRI tensors and save them
        for i in range(3):  # 3 subjects
            subject_id = f"sub-A10{i}"  # Matches the regex
            file_path = Path(self.temp_dir.name) / f"{subject_id}.pt"
            tensor_data = torch.rand((10, 128, 128))  # 10 slices, 128x128 each
            torch.save(tensor_data, file_path)
            self.mri_files.append(str(file_path))

        # Create a fake labels DataFrame
        self.labels_df = pd.DataFrame({
            "participant_id": ["A100", "A101", "A102"],
            "dx_encoded": [0, 1, 1]  # Example labels
        })

        # Initialize dataset
        self.dataset = MRIDataset(self.mri_files, self.labels_df, target_shape=(224, 224))

    def tearDown(self):
        self.temp_dir.cleanup()  # Clean up temp files

    def test_dataset_length(self):
        # Each file has 10 slices, 3 files total = 30 slices
        self.assertEqual(len(self.dataset), 30)

    def test_getitem_valid(self):
        # Retrieve a sample
        slice_data, label = self.dataset[5]

        # Check tensor properties
        self.assertIsInstance(slice_data, torch.Tensor)
        self.assertEqual(slice_data.shape, (1, 224, 224))  # Resized to target shape
        self.assertIsInstance(label, int)

    def test_getitem_out_of_range(self):
        with self.assertRaises(IndexError):
            self.dataset[100]  # Out of bounds

    def test_missing_label(self):
        # Create dataset with a missing label case
        labels_df_missing = pd.DataFrame({
            "participant_id": ["A100", "A101"],  # Missing "A102"
            "dx_encoded": [0, 1]
        })
        dataset_missing_label = MRIDataset(self.mri_files, labels_df_missing)

        with self.assertRaises(ValueError):
            dataset_missing_label[25]  # This slice belongs to A102, which has no label

    def test_extract_subject_number(self):
        # Test subject number extraction
        self.assertEqual(self.dataset._extract_subject_number("sub-A123.nii.gz"), "A123")
        self.assertEqual(self.dataset._extract_subject_number("randomfile.nii.gz"), None)

if __name__ == "__main__":
    unittest.main()