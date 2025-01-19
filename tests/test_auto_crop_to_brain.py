import os
import sys
import numpy as np
import pandas as pd

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocess import load_nii, get_data, resample_image, normalize_data, extract_brain, auto_crop_to_brain
from src.utils.preprocess_validation import plot_slices, calculate_snr, calculate_mse, calculate_psnr, calculate_ssim

def main():
    pass
    

if __name__ == '__main__':
    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path_t1 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    rel_file_path_t2 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_T2w.nii.gz"
    file_path = os.path.join(current_dir, '..', rel_file_path_t1)

    nii = load_nii(rel_file_path_t1)
    resampled_data = resample_image(nii, order=1, mode='wrap', cval=0.0)
    normalized_data = normalize_data(resampled_data)
    #original_data = get_data(nii)
    extracted_brain, mask = extract_brain(normalized_data, modality='t1', return_mask=True, mask_output_type='numpy')
    cropped_data = auto_crop_to_brain(extracted_brain, mask, pad_shape=None)
    plot_slices(extracted_brain, how_many=8, title="Extracted Brain")
    plot_slices(cropped_data, how_many=8, title="Cropped Brain")


