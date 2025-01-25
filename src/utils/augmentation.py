import torch
import random
from torchvision import transforms

# TODO: finalize the augmentation pipeline
# TODO: parameters should be configurable?

# From the literature, the following augmentations are justified and can be used:
# - Random horizontal flip (brain is symmetric along the sagittal plane)
# - Random vertical flip (brain is symmetric along the coronal plane), less common
# - Random rotation (useful for handling minor variations in patient positioning during scans)
#   - Rotation angle should be relatively small, e.g. 10-30 degrees; larger rotations can lead to unrealistic data
# - Random affine transformation (useful for handling minor variations in patient positioning during scans)
#   - Translation: small shifts in the x and y directions (e.g. 10% of the image size)
#   - Shear: small shearing angles (e.g. 10-20 degrees)
#   - Scale: small scaling factors (e.g. 0.8-1.2)
# - Random intensity shift (useful for handling variations in scanner settings)
#   - Random brightness or random gamma
#   - Gaussian noise (useful to simulate variations and imperfections in the data)
# - Elastic deformation (useful to simulate local deformations in the brain) TODO: add this?
# - Cropping and resizing

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if random.random() < 0.5:  # 50% chance of applying
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor

class RandomIntensityShift:
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, tensor):
        if random.random() < 0.5:  # 50% chance of applying
            factor = random.uniform(1 - self.shift_range, 1 + self.shift_range)
            return tensor * factor
        return tensor

class AdjustContrast3D:
    def __init__(self, contrast_factor=0.1):
        self.contrast_factor = contrast_factor

    def __call__(self, tensor):
        # Simulate a random contrast factor around the base contrast_factor
        factor = 1 + (torch.rand(1).item() * 2 - 1) * self.contrast_factor
        mean = tensor.mean()
        return (tensor - mean) * factor + mean


def get_augmentation():
    """
    Get the augmentation pipeline for the data
    
    Returns:
    torchvision.transforms.Compose: augmentation pipeline
    """
    augmentations = transforms.Compose([
        transforms.ToTensor(),  # Convert NumPy array to PyTorch tensor
        #transforms.Resize((128, 128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=10, 
                                translate=(0.1, 0.1),
                                scale=(0.9, 1.1),
                                shear=10
                                ),
        AdjustContrast3D(contrast_factor=0.1), # TODO: deprecate?
        AddGaussianNoise(mean=0, std=0.01),
        RandomIntensityShift(shift_range=0.1),
        transforms.Normalize(mean=[0.5], std=[0.5]) # previously normalized data to [0, 1]                        
    ])
    return augmentations
