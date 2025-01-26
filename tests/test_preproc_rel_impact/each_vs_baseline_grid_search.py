import os
import sys
import itertools
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
from src.utils.preprocess_validation import calculate_cnr, calculate_ssim
from src.utils.preprocess_validation import calculate_relative_psnr, calculate_relative_rmse
#from src.utils.preprocess_validation import plot_slices

def calculate_metrics(reference_data, modified_data, preprocessing_step):

    results_metrics = []
    # Compute metrics
    modified_data_matched = match_dimensions(reference_data, modified_data)

    snr_value = calculate_snr_with_mask(modified_data_matched)
    cnr_value = calculate_cnr(modified_data_matched)  

    mse_value = calculate_mse(reference_data, modified_data_matched)
    psnr_value = calculate_psnr(reference_data, modified_data_matched, mse=mse_value)
    ssim_value = calculate_ssim(reference_data, modified_data_matched)

    relative_psnr_value = calculate_relative_psnr(modified_data_matched)
    relative_rmse_value = calculate_relative_rmse(modified_data_matched)

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
    # Define parameter grid
    orders = range(0, 6)  # Spline interpolation orders
    modes = ['constant', 'nearest', 'reflect', 'wrap']  # Boundary handling modes
    cvals = [0.0, 0.5, 1.0]  # Constant fill values
    voxel_size = (1, 1, 1)  # Fixed for this example

    # Create all parameter combinations
    param_grid = list(itertools.product(orders, modes, cvals))
    grid_resampling_results = []

    combination_count = 0
    # Iterate through parameter combinations
    for order, mode, cval in param_grid:
        try:
            print(f'Current combination count: {combination_count}/{len(param_grid)}')
            # Resample the image    
            resampled_image = resample_image(nii, voxel_size=voxel_size, order=order, mode=mode, cval=cval)
            resampling_metrics = calculate_metrics(original_data, resampled_image, step)
            params = {'order': order, 'mode': mode, 'cval': cval}
            combined_metrics_params = {**resampling_metrics, **params}
            grid_resampling_results.append(combined_metrics_params)

            combination_count += 1
            #if combination_count == 2:
            #    break
        except Exception as e:
            # Handle any errors during resampling or SNR calculation
            print(f"Error with combination (order={order}, mode={mode}, cval={cval}): {str(e)}")
            continue    

    # Find the best combination# Find the best combination
    grid_resampling_best_results = max(grid_resampling_results, key=lambda x: x['snr'])

    pd.DataFrame(grid_resampling_results).to_csv('grid_search_resampling_metrics_each_vs_baseline_all.csv', index=False)
    pd.DataFrame([grid_resampling_best_results]).to_csv('grid_search_resampling_metrics_each_vs_baseline_best.csv', index=False)

    # Append values of the best combination to the resampling function and calculate the metric again
    resampled_image = resample_image(nii, 
                                     voxel_size=voxel_size, 
                                     order=grid_resampling_best_results['order'], 
                                     mode=grid_resampling_best_results['mode'], 
                                     cval=grid_resampling_best_results['cval'])
    resampling_metrics = calculate_metrics(original_data, resampled_image, step)
    plot_save_selected_slices(resampled_image, step=step, isSave_slices=True)

    # Stage 2
    step = 'normalization'
    normalized_data = normalize_data(resampled_image)
    normalization_metrics = calculate_metrics(original_data, normalized_data, step)
    plot_save_selected_slices(normalized_data, step=step, isSave_slices=True)

    # Stage 3
    step = 'brain_extraction'
    brain_data = extract_brain(normalized_data, what_to_return={'extracted_brain': 'numpy', 'mask': 'numpy'})
    extracted_brain = brain_data['extracted_brain']
    brain_extraction_metrics = calculate_metrics(original_data, extracted_brain, step)
    plot_save_selected_slices(extracted_brain, step=step, isSave_slices=True)

    # Stage 4
    step = 'cropping'
    mask = brain_data['mask']
    cropped_data = crop_to_largest_bounding_box(extracted_brain, mask=mask)
    cropping_metrics = calculate_metrics(original_data, cropped_data, step)
    plot_save_selected_slices(cropped_data, step=step, isSave_slices=True)

    # Stage 5
    step = 'smoothing'
    # Define parameter grid
    sigmas = [0.5, 1.0, 1.5, 2.0]  # Standard deviation of the Gaussian kernel
    truncations = [2.0, 3.0, 4.0]  # Truncation factor for the kernel size
    modes = ['constant', 'reflect', 'nearest', 'wrap']  # Boundary handling modes
    cvals = [0.0, 0.5, 1.0]  # Constant fill values
    orders = [0]  # Order of the filter (0 = smoothing, 1 = first derivative, etc.) # 1 and 2 makes no sense

    # Create all parameter combinations
    param_grid = list(itertools.product(sigmas, truncations, modes, cvals, orders))
    grid_smoothing_results = []

    combination_count = 0
    for sigma, truncate, mode, cval, order in param_grid:
        try:
            print(f'Current combination: {combination_count + 1}/{len(param_grid)}')
            # Apply Gaussian smoothing
            smoothed_data = apply_gaussian_smoothing(cropped_data, sigma=sigma, order=order, mode=mode, cval=cval, truncate=truncate)
            smoothing_metrics = calculate_metrics(original_data, smoothed_data, step)
            params = {'sigma': sigma, 'truncate': truncate,'mode': mode, 'cval': cval, 'order': order}
            combined_metrics_params = {**smoothing_metrics, **params}
            grid_smoothing_results.append(combined_metrics_params)

            combination_count += 1
            #if combination_count == 2:
            #    break
        except Exception as e:
            print(f"Error with combination (sigma={sigma}, truncate={truncate}, mode={mode}, cval={cval}, order={order}): {str(e)}")
            continue

    # Find the best combination
    grid_smoothing_best_results = max(grid_smoothing_results, key=lambda x: x['snr'])

    pd.DataFrame(grid_smoothing_results).to_csv('grid_search_smoothing_metrics_each_vs_baseline_all.csv', index=False)
    pd.DataFrame([grid_smoothing_best_results]).to_csv('grid_search_smoothing_metrics_each_vs_baseline_best.csv', index=False)
    
    # Append values of the best combination to the resampling function and calculate the metric again
    smoothed_data = apply_gaussian_smoothing(cropped_data, 
                                             sigma=grid_smoothing_best_results['sigma'], 
                                             order=grid_smoothing_best_results['order'], 
                                             mode=grid_smoothing_best_results['mode'], 
                                             cval=grid_smoothing_best_results['cval'], 
                                             truncate=grid_smoothing_best_results['truncate'])
    smoothing_metrics = calculate_metrics(original_data, smoothed_data, step)
    plot_save_selected_slices(smoothed_data, step=step, isSave_slices=True)

    # Stage 6
    step = 'normalized_smoothing'
    smoothed_normalized_data = normalize_data(smoothed_data)
    normalized_smoothing_metrics = calculate_metrics(original_data, smoothed_normalized_data, step)
    plot_save_selected_slices(smoothed_normalized_data, step=step, isSave_slices=True)

    all_results = [raw_metrics, 
                   resampling_metrics, 
                   normalization_metrics, 
                   brain_extraction_metrics, 
                   cropping_metrics, 
                   smoothing_metrics,
                   normalized_smoothing_metrics]

    return all_results

def plot_save_selected_slices(data_at_step, step: str, isSave_slices=False):

    # Plot the slices
    fig, ax = plt.subplots(1, 1, figsize=(1.92*1.5, 2.56*1.5))
    ax.imshow(data_at_step[:, :, 140], cmap="gray")
    ax.axis("off")
    #ax.set_title(title)
    if isSave_slices:
        save_fig_to = f'grid_search_data_at_step_{step}.png'
        plt.savefig(save_fig_to, bbox_inches='tight')

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

    path_to_all_results = 'grid_search_metrics_each_vs_baseline.csv'
    results_df.to_csv(path_to_all_results, index=False)
