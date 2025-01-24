import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.preprocess import load_nii, get_data, match_dimensions 
from src.utils.preprocess import resample_image, normalize_data
from src.utils.preprocess import extract_brain
from src.utils.preprocess import get_largest_brain_mask_slice, crop_to_largest_bounding_box
from src.utils.preprocess import apply_gaussian_smoothing

from src.utils.preprocess_validation import plot_slices, calculate_mse
from src.utils.preprocess_validation import calculate_snr_with_mask, calculate_psnr
from src.utils.preprocess_validation import calculate_mse, calculate_ssim

from src.utils.augmentation import AddGaussianNoise

from torchvision import transforms


def complete_preprocess(path_to_file: str):

    loaded_nii = load_nii(path_to_file)

    return preprocessed_image

def augment_data(preprocessed_data, mean=mean, std=std):

    augmentations = transforms.Compose([AddGaussianNoise(mean=mean, std=std)])

    return augmented_image

def calculate_metrics(reference_image, image_to_compare):
    pass


if __name__ == "__main__":
    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    file_path = os.path.join(current_dir, '..', "..", rel_file_path)

    preprocessed_data = complete_preprocess(file_path)

    augmented_data = augment_data(preprocessed_data)

    metrics = calculate_metrics(preprocessed_data, augmented_data)
