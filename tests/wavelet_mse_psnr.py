import os
import sys
import itertools
import numpy as np
import pandas as pd

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocess import load_nii, get_data, apply_wavelet_denoising, match_dimensions
from src.utils.preprocess_validation import calculate_mse, calculate_psnr


def grid_search_wavelet_mse_psnr(file_path):
    """
    Perform a grid search to find the best combination of Wavelet denoising parameters 
    that yield the lowest MSE.

    Args:
        original_data (numpy.ndarray): 3D MRI data.

    Returns:
        dict: Best parameter combination and corresponding MSE.
    """

    # Load the NIfTI file
    nii = load_nii(file_path)
    original_data = get_data(nii)  # Extract data from the NIfTI object

    # Define parameter grid
    wavelet = ['db1', 'sym4', 'coif1']  # Different wavelet types
    level = [2, 3, 4]  # Levels of wavelet decomposition
    threshold = [0.01, 0.03, 0.05]  # Threshold values for soft/hard thresholding
    thresholding_mode = ['soft', 'hard']  # Thresholding types
    boundary_mode = ['symmetric', 'reflect', 'constant']  # Boundary handling modes

    # Create all parameter combinations
    param_grid = list(itertools.product(wavelet, level, threshold, thresholding_mode, boundary_mode))
    results = []

    combination_count = 0
    for wavelet, level, threshold, thresholding_mode, boundary_mode in param_grid:
        try:
            print(f'Current combination: {combination_count + 1}/{len(param_grid)}')
            # Apply Wavelet denoising
            wavelet_data = apply_wavelet_denoising(original_data, wavelet=wavelet, level=level, threshold=threshold, thresholding=thresholding_mode, mode=boundary_mode)
            
            # Match dimensions if necessary (in case of boundary effects)
            wavelet_data_matched = match_dimensions(original_data, wavelet_data)
            
            # Calculate MSE and PSNR
            mse = calculate_mse(original_data, wavelet_data_matched)
            psnr = calculate_psnr(original_data, wavelet_data_matched, mse)

            # Add result to the list of results
            results.append({
                'wavelet': wavelet,
                'level': level,
                'threshold': threshold,
                'thresholding': thresholding_mode,
                'mode': boundary_mode,
                'mse': mse,
                'psnr': psnr
            })
            combination_count += 1
        except Exception as e:
            print(f"Error with combination (wavelet={wavelet}, level={level}, threshold={threshold}, thresholding={thresholding_mode}, mode={boundary_mode}): {str(e)}")
            continue

    # Find the best combination
    best_result = min(results, key=lambda x: x['mse'])
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
    # Run grid search for Wavelet denoising
    best_params, all_results = grid_search_wavelet_mse_psnr(file_path)

    # Save results
    output_file_name = "wavelet_mse_psnr_results.csv"
    best_output_file_name = "wavelet_mse_psnr_best.csv"
    save_results(best_params, all_results, output_file_name, best_output_file_name)

    print("Best Parameters:")
    print(best_params)

    print("\nAll Results:")
    for result in all_results:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})