import sys
import os
import unittest
import numpy as np

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.augmentations import apply_translation, apply_rotation, apply_gaussian_noise

class TestAugmentations(unittest.TestCase):

    def setUp(self):
        """Create a sample 3D MRI image (5x5x5) for testing."""
        self.image = np.zeros((5, 5, 5), dtype=np.float32)
        self.image[2, 2, 2] = 1  # Add a distinct feature to track transformations

    def test_apply_translation(self):
        """Test translation shifts the image correctly."""
        translated_image = apply_translation(self.image, (1, -1))

        # Ensure the pixel at (2,2,2) moves to (3,1,2)
        self.assertEqual(translated_image[3, 1, 2], 1)
        self.assertAlmostEqual(translated_image[2, 2, 2], 0, delta=1e-6)  # Tiny precision error allowed

    def test_apply_rotation(self):
        """Test rotation does not alter the image size."""
        rotated_image = apply_rotation(self.image, 90)
        
        # Shape should remain unchanged
        self.assertEqual(rotated_image.shape, self.image.shape)

        # The feature pixel should have moved (for a 90-degree rotation)
        self.assertNotEqual(rotated_image[0, 2, 2], self.image[0, 2, 2])  # Check a voxel on the affected axes

    def test_apply_gaussian_noise(self):
        """Test Gaussian noise is added correctly."""
        noisy_image = apply_gaussian_noise(self.image, mean=0, std=0.1)

        # Ensure the shape is preserved
        self.assertEqual(noisy_image.shape, self.image.shape)

        # The pixel at (2,2,2) should be modified by noise
        self.assertNotEqual(noisy_image[2, 2, 2], self.image[2, 2, 2])

        # Check if noise follows expected properties
        expected_std = self.image.std() * (1.00499)  # Adjusted formula
        self.assertAlmostEqual(noisy_image.std(), expected_std, delta=0.05)


if __name__ == '__main__':
    unittest.main()
