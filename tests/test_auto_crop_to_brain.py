import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocess import load_nii, get_data, resample_image, normalize_data, match_dimensions, extract_brain
from src.utils.preprocess import get_largest_brain_mask_slice, crop_to_largest_bounding_box
from src.utils.preprocess_validation import plot_slices, calculate_snr, calculate_mse, calculate_psnr, calculate_ssim

def plot_comparison(extracted_brain: np.ndarray, cropped_slices: np.ndarray) -> None:
    """
    Plot slices of the extracted brain and cropped brain for comparison
    
    Args:
    - extracted_brain (np.ndarray): Extracted brain data.
    - cropped_slices (np.ndarray): Cropped brain slices.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))  # 2 rows, 4 columns
    plot_slices(extracted_brain, how_many=4, title="Extracted Brain", axes=axes[0])
    plot_slices(cropped_slices, how_many=4, title="Cropped Brain", axes=axes[1])
    fig.suptitle("Slice Comparison")
    plt.show()

def main(file_path: str, preprocess: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the MRI data, preprocess it if necessary, and extract the brain.
    
    Args:
    - file_path (str): Path to the MRI file.
    - preprocess (bool): Whether to preprocess the MRI data (True) or not (False).
    
    Returns:
    - extracted_brain (np.ndarray): Extracted brain slices
    - cropped_slices (np.ndarray): Cropped brain slices."""

    nii = load_nii(file_path)
    
    if preprocess:
        resampled_data = resample_image(nii, order=1, mode='wrap', cval=0.0)
        normalized_data = normalize_data(resampled_data)
        data = normalized_data  # Replace the original data with the normalized one for further processing
    else:
        data = get_data(nii)

    brain_extraction_result = extract_brain(data, 
                                            modality='t1', 
                                            what_to_return={'extracted_brain': 'numpy', 'mask':'numpy'})
    
    extracted_brain = brain_extraction_result['extracted_brain']
    mask = brain_extraction_result['mask']
    binary_mask, largest_slice_index = get_largest_brain_mask_slice(mask)
    cropped_slices = crop_to_largest_bounding_box(extracted_brain, binary_mask, largest_slice_index)
    return extracted_brain, cropped_slices

if __name__ == '__main__':
    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path_t1 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    rel_file_path_t2 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_T2w.nii.gz"
    file_path = os.path.join(current_dir, '..', rel_file_path_t1)

    do_preprocess = False
    extracted_brain, cropped_slices = main(file_path, do_preprocess)

    compare_slices = False
    if compare_slices:
        plot_comparison(extracted_brain, cropped_slices)

    # Calculate metrics
    snr_value = calculate_snr(cropped_slices)
    cropped_slices_matched = match_dimensions(extracted_brain, cropped_slices)
    mne_value = calculate_mse(extracted_brain, cropped_slices_matched)
    psnr_value = calculate_psnr(extracted_brain, cropped_slices_matched, mse=mne_value)
    ssim_value = calculate_ssim(extracted_brain, cropped_slices_matched)

    # Printing the metrics
    print(f'SNR: {snr_value}\nMNE: {mne_value}\nPSNR: {psnr_value}\nSSIM: {ssim_value}\n')

    # Saving results to CSV file
    metrics_df = pd.DataFrame(
        {
            "SNR": [snr_value],
            "MSE": [mne_value],
            "PSNR": [psnr_value],
            "SSIM": [ssim_value]
        }
    )

    if do_preprocess:
        mode = 'processed'
    else:
        mode = 'clean'
         
    metrics_df.to_csv(os.path.join(current_dir, f"auto_crop_metrics_{mode}.csv"), index=False)
