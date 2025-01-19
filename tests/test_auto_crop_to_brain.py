import os
import sys
#import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocess import load_nii, resample_image, normalize_data, extract_brain
from src.utils.preprocess import get_largest_brain_mask_slice, crop_to_largest_bounding_box
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
    #extracted_brain, mask = extract_brain(normalized_data, modality='t1', return_mask=True, mask_output_type='numpy')
    #cropped_data = auto_crop_to_brain(extracted_brain, mask, pad_shape=None)
    #plot_slices(extracted_brain, how_many=8, title="Extracted Brain")
    #plot_slices(cropped_data, how_many=8, title="Cropped Brain")

    brain_extraction_result = extract_brain(normalized_data, 
                                            modality='t1', 
                                            what_to_return={'extracted_brain': 'numpy', 'mask':'numpy'})
    
    extracted_brain = brain_extraction_result['extracted_brain']
    mask = brain_extraction_result['mask']
    binary_mask, largest_slice_index = get_largest_brain_mask_slice(mask)
    cropped_slices = crop_to_largest_bounding_box(extracted_brain, binary_mask, largest_slice_index)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))  # 2 rows, 4 columns
    plot_slices(extracted_brain, how_many=4, title="Extracted Brain", axes=axes[0])
    plot_slices(cropped_slices, how_many=4, title="Cropped Brain", axes=axes[1])
    fig.suptitle("Slice Comparison")
    plt.show()

# TODO: calculate metrics


