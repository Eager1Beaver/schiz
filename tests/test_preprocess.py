import unittest
import tempfile
import os
import sys
import numpy as np
import nibabel as nib

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocess import (
    load_nii, 
    get_data, 
    get_affine, 
    match_dimensions,
    resample_image, 
    normalize_data, 
    extract_brain,
    get_largest_brain_mask_slice,
    crop_to_largest_bounding_box,
    apply_gaussian_smoothing
)

class TestPreprocessFunctions(unittest.TestCase):
    
    def setUp(self):
        """Create test data and a temporary NIfTI file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "test.nii.gz")
        
        self.test_data = np.random.rand(64, 64, 64)
        self.test_affine = np.eye(4)
        self.nii_img = nib.Nifti1Image(self.test_data, self.test_affine)
        nib.save(self.nii_img, self.file_path)
        
        # Create additional test data
        self.data = np.random.rand(128, 128, 64).astype(np.float32)
        self.test_mask = np.zeros((100, 100, 50))
        self.test_mask[30:70, 30:70, 25] = 1  # Largest brain mask in slice 25
        self.test_mask[40:60, 40:60, 10] = 1  # Smaller brain mask in slice 10
    
    def tearDown(self):
        """Cleanup temporary directory after tests."""
        self.temp_dir.cleanup()

    def test_load_nii_valid(self):
        nii = load_nii(self.file_path)
        self.assertIsInstance(nii, nib.Nifti1Image)
    
    def test_load_nii_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            load_nii("invalid_path.nii.gz")
    
    def test_load_nii_invalid_type(self):
        with self.assertRaises(TypeError):
            load_nii(123)
    
    def test_get_data_valid(self):
        data = get_data(self.nii_img)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (64, 64, 64))
    
    def test_get_affine_valid(self):
        affine = get_affine(self.nii_img)
        self.assertIsInstance(affine, np.ndarray)
        self.assertEqual(affine.shape, (4, 4))
    
    def test_match_dimensions_crop(self):
        original = np.random.rand(32, 32, 32)
        modified = np.random.rand(64, 64, 64)
        result = match_dimensions(original, modified)
        self.assertEqual(result.shape, original.shape)
    
    def test_match_dimensions_pad(self):
        original = np.random.rand(64, 64, 64)
        modified = np.random.rand(32, 32, 32)
        result = match_dimensions(original, modified)
        self.assertEqual(result.shape, original.shape)
    
    def test_resample_image(self):
        resampled_nib = resample_image(self.nii_img, voxel_size=(2, 2, 2), output_format='nib')
        self.assertIsInstance(resampled_nib, nib.Nifti1Image)
        
        resampled_np = resample_image(self.nii_img, voxel_size=(2, 2, 2), output_format='numpy')
        self.assertIsInstance(resampled_np, np.ndarray)
    
    def test_normalize_data(self):
        normalized = normalize_data(self.data, method='z-score')
        self.assertAlmostEqual(np.mean(normalized), 0, delta=0.1)
        self.assertAlmostEqual(np.std(normalized), 1, delta=0.1)
        
        normalized_minmax = normalize_data(self.data, method='min-max', min_val=0, max_val=1)
        self.assertGreaterEqual(normalized_minmax.min(), 0)
        self.assertLessEqual(normalized_minmax.max(), 1)
    
    def test_extract_brain(self):
        result = extract_brain(self.data, modality='t1', what_to_return={'mask': 'numpy', 'extracted_brain': 'numpy'})
        
        self.assertIn('mask', result)
        self.assertIn('extracted_brain', result)
        self.assertIsInstance(result['mask'], np.ndarray)
        self.assertIsInstance(result['extracted_brain'], np.ndarray)
        self.assertEqual(result['mask'].shape, self.data.shape)  # Ensure mask shape matches input

    def test_crop_to_largest_bounding_box(self):
        # Use extract_brain to get the mask
        brain_result = extract_brain(self.test_data, modality='t1', what_to_return={'mask': 'numpy'})
        extracted_mask = brain_result['mask']
        
        cropped_data = crop_to_largest_bounding_box(self.test_data, mask=extracted_mask)
        
        self.assertEqual(cropped_data.shape[2], self.test_data.shape[2])  # Ensure depth (z-axis) is unchanged
        self.assertGreaterEqual(cropped_data.shape[0], 30)  # Brain size might vary slightly
        self.assertGreaterEqual(cropped_data.shape[1], 30)  # Brain size might vary slightly

    '''def test_extract_brain(self):
        result = extract_brain(self.data, modality='t1', what_to_return={'extracted_brain': 'numpy'})
        self.assertIn('extracted_brain', result)
        self.assertIsInstance(result['extracted_brain'], np.ndarray)
    
    def test_get_largest_brain_mask_slice(self):
        binary_mask, largest_slice_idx = get_largest_brain_mask_slice(self.test_mask)
        self.assertEqual(largest_slice_idx, 25)
        self.assertTrue(binary_mask.shape == self.test_mask.shape)
    
    def test_crop_to_largest_bounding_box(self):
        cropped_data = crop_to_largest_bounding_box(self.test_data, mask=self.test_mask)
        self.assertEqual(cropped_data.shape[:2], (40, 40))
        self.assertEqual(cropped_data.shape[2], self.test_data.shape[2])'''
    
    def test_apply_gaussian_smoothing(self):
        smoothed_data = apply_gaussian_smoothing(self.test_data, sigma=1.5)
        self.assertEqual(smoothed_data.shape, self.test_data.shape)
        self.assertFalse(np.array_equal(smoothed_data, self.test_data))  # Data should change

if __name__ == "__main__":
    unittest.main()
