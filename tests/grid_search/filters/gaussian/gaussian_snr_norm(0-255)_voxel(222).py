import os
import sys
import itertools
import numpy as np
import pandas as pd

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from src.utils.preprocess import load_nii, resample_image, normalize_data, apply_gaussian_smoothing
from src.utils.preprocess_validation import calculate_snr_with_mask


def grid_search_smoothing_snr(file_path):
    """
    Perform a grid search to find the best combination of Gaussian smoothing parameters 
    that yield the highest SNR.

    Args:
        original_data (numpy.ndarray): 3D MRI data.

    Returns:
        dict: Best parameter combination and corresponding SNR.
    """

    # Load the NIfTI file
    nii = load_nii(file_path)
    resampled_data = resample_image(nii)
    normalized_data = normalize_data(resampled_data)

    # Define parameter grid
    sigmas = [0.5, 1.0, 1.5, 2.0]  # Standard deviation of the Gaussian kernel
    truncations = [2.0, 3.0, 4.0]  # Truncation factor for the kernel size
    modes = ['constant', 'reflect', 'nearest', 'wrap']  # Boundary handling modes
    cvals = [0.0, 0.5, 1.0]  # Constant fill values
    orders = [0, 1, 2]  # Order of the filter (0 = smoothing, 1 = first derivative, etc.)

    # Create all parameter combinations
    param_grid = list(itertools.product(sigmas, truncations, modes, cvals, orders))
    results = []

    combination_count = 0
    for sigma, truncate, mode, cval, order in param_grid:
        try:
            print(f'Current combination: {combination_count + 1}/{len(param_grid)}')
            # Apply Gaussian smoothing
            smoothed_data = apply_gaussian_smoothing(normalized_data, sigma=sigma, order=order, mode=mode, cval=cval, truncate=truncate)
            
            final_data = normalize_data(smoothed_data)

            # Calculate SNR
            snr_val = calculate_snr_with_mask(final_data)

            # Add result to the list of results
            results.append({
                'sigma': sigma,
                'truncate': truncate,
                'mode': mode,
                'cval': cval,
                'order': order,
                'snr': snr_val
            })
            combination_count += 1
        except Exception as e:
            print(f"Error with combination (sigma={sigma}, truncate={truncate}, mode={mode}, cval={cval}, order={order}): {str(e)}")
            continue

    # Find the best combination
    best_result = max(results, key=lambda x: x['snr'])
    return best_result, results

def save_results(best_params, all_results, output_file_name, best_output_file_name):
    """
    Save the best parameters and all results to CSV files.
    """
    # Convert all_results (list of dictionaries) into a DataFrame
    all_results_df = pd.DataFrame(all_results)
    # Save all results to CSV
    all_results_df.to_csv(output_file_name, index=False)

    # Save best_params separately
    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv(best_output_file_name, index=False)

    print(f"Results saved to {output_file_name}")
    print(f'Best params saved to {best_output_file_name}')


if __name__ == "__main__":
    file_path = r"C:\Users\Dell\Downloads\sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    # Run grid search for Gaussian smoothing
    best_params, all_results = grid_search_smoothing_snr(file_path)

    # Save results
    output_file_name = "gaussian_snr_results_norm(0-255)_voxel(222).csv"
    best_output_file_name = "gaussian_snr_best_norm(0-255)_voxel(222).csv"
    save_results(best_params, all_results, output_file_name, best_output_file_name)

    print("Best Parameters:")
    print(best_params)

    print("\nAll Results:")
    for result in all_results:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})