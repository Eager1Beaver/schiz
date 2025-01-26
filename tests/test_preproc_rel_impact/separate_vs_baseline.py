import os
import sys
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..')))

from src.utils.preprocess import load_nii, get_data 
from src.utils.preprocess import resample_image, normalize_data
from src.utils.preprocess import match_dimensions

from src.utils.preprocess import extract_brain 
from src.utils.preprocess import crop_to_largest_bounding_box
from src.utils.preprocess import apply_gaussian_smoothing

from src.utils.preprocess_validation import calculate_snr_with_mask
from src.utils.preprocess_validation import calculate_mse, calculate_psnr
from src.utils.preprocess_validation import calculate_cnr, calculate_ssim#, generate_signal_mask
from src.utils.preprocess_validation import calculate_relative_psnr, calculate_relative_rmse
#from src.utils.preprocess_validation import plot_slices

def calculate_metrics(reference_data, modified_data, preprocessing_step):

    results_metrics = []
    # Compute metrics
    modified_data_matched = match_dimensions(reference_data, modified_data)
    
    snr_value = calculate_snr_with_mask(modified_data)
    #signal_mask = generate_signal_mask(modified_data_matched)
    cnr_value = calculate_cnr(modified_data)  

    mse_value = calculate_mse(reference_data, modified_data_matched)
    psnr_value = calculate_psnr(reference_data, modified_data_matched, mse=mse_value)
    ssim_value = calculate_ssim(reference_data, modified_data_matched)

    relative_psnr_value = calculate_relative_psnr(modified_data)
    relative_rmse_value = calculate_relative_rmse(modified_data)

    results_metrics = {'preprocessing_step': preprocessing_step,
                        'snr': snr_value,
                        'cnr': cnr_value,
                        'mse': mse_value,
                        'psnr': psnr_value,
                        'ssim': ssim_value,
                        'relative_psnr': relative_psnr_value,
                        'relative_rmse': relative_rmse_value}
    
    return results_metrics

def assess_metrics(file_path):
    
    all_results = []

    # Stage 0
    step = 'raw'
    nii = load_nii(file_path)
    original_data = get_data(nii)
    #print('original after loading ', original_data.max(), original_data.min(), original_data.mean(), original_data.std(), original_data.shape)
    raw_metrics = {'preprocessing_step': step,
                   'snr': 13.0817,
                   'cnr': 12.3246,
                   'mse': 0,
                   'psnr': 0,
                   'ssim': 1,
                   'relative_psnr': 20.8055,
                   'relative_rmse': 875.427}
    
    plot_save_selected_slices(original_data, step=step, isSave_slices=True)

    # Stage 1
    step = 'resampling'
    resampled_image = resample_image(nii, voxel_size=(1,1,1), order=0, mode='wrap', cval=0)
    resampling_metrics = calculate_metrics(original_data, resampled_image, step)
    plot_save_selected_slices(resampled_image, step=step, isSave_slices=True)
    #print(f'metric after resaple: {resampling_metrics}')

    # Stage 2
    step = 'normalization'
    normalized_data = normalize_data(original_data)
    normalization_metrics = calculate_metrics(original_data, normalized_data, step)
    plot_save_selected_slices(normalized_data, step=step, isSave_slices=True)

    # Stage 3
    step = 'brain_extraction'
    brain_data = extract_brain(original_data, what_to_return={'extracted_brain': 'numpy', 'mask': 'numpy'})
    extracted_brain = brain_data['extracted_brain']
    brain_extraction_metrics = calculate_metrics(original_data, extracted_brain, step)
    plot_save_selected_slices(extracted_brain, step=step, isSave_slices=True)

    # Stage 4
    step = 'cropping'
    mask = brain_data['mask']
    cropped_data = crop_to_largest_bounding_box(original_data, mask=mask)
    cropping_metrics = calculate_metrics(original_data, cropped_data, step)
    plot_save_selected_slices(cropped_data, step=step, isSave_slices=True)

    # Stage 5
    step = 'smoothing'
    #print('original before smoothing ', original_data.max(), original_data.min(), original_data.mean(), original_data.std(), original_data.shape)
    smoothed_data = apply_gaussian_smoothing(original_data, sigma=0.5, order=0, mode='reflect', cval=0, truncate=2)
    #print('smoothing ', smoothed_data.max(), smoothed_data.min(), smoothed_data.mean(), smoothed_data.std(), smoothed_data.shape)
    smoothing_metrics = calculate_metrics(original_data, smoothed_data, step)
    plot_save_selected_slices(smoothed_data, step=step, isSave_slices=True)

    # Stage 6
    #step = 'normalized_smoothing'
    #smoothed_normalized_data = normalize_data(original_data)
    #normalized_smoothing_metrics = calculate_metrics(original_data, smoothed_normalized_data, step)
    #plot_save_selected_slices(smoothed_normalized_data, step=step, isSave_slices=True)

    all_results = [raw_metrics, 
                   resampling_metrics, 
                   normalization_metrics, 
                   brain_extraction_metrics, 
                   cropping_metrics, 
                   smoothing_metrics#,
                   #normalized_smoothing_metrics
                   ]

    return all_results

def plot_save_selected_slices(data_at_step, step: str, isSave_slices=False):

    # Plot the slices
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(data_at_step[:, :, 140], cmap="gray")
    ax.axis("off")
    #ax.set_title(title)
    if isSave_slices:
        save_fig_to = f'data_at_step_{step}_separate.png'
        plt.savefig(save_fig_to)

    print(f"Selected slices saved to {save_fig_to}")

if __name__ == "__main__":
    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    #rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000368/ses-20110101/anat/sub-A00000368_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    rel_file_path = 'data/schizconnect_COBRE_images_22613/COBRE/sub-A00014839/ses-20090101/anat/sub-A00014839_ses-20090101_acq-mprage_run-02_T1w.nii.gz'

    file_path = os.path.join(current_dir, '..', '..', rel_file_path)

    all_results = assess_metrics(file_path)

    results_df = pd.DataFrame(all_results)
    # Display the DataFrame
    print(results_df)

    path_to_all_results = 'metrics_separate_vs_baseline2.csv'
    results_df.to_csv(path_to_all_results, index=False)
