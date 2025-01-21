import os
import sys
import itertools
import numpy as np
import pandas as pd

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocess import load_nii, get_data, apply_median_filter, match_dimensions
from src.utils.preprocess_validation import calculate_ssim


def grid_search_median_ssim(file_path):
    """
    Perform a grid search to find the best combination of Median filtering parameters 
    that yield the highest Structural Similarity Index (SSIM).

    Args:
        original_data (numpy.ndarray): 3D MRI data.

    Returns:
        dict: Best parameter combination and corresponding SSIM.
    """

    # Load the NIfTI file
    nii = load_nii(file_path)
    original_data = get_data(nii)  # Extract data from the NIfTI object

    # Define parameter grid
    filter_size = [(3, 3, 3), (5, 5, 5), (3, 3, 7)]  # Neighborhood sizes (isotropic and anisotropic)
    mode = ['constant', 'reflect', 'nearest', 'wrap']  # Boundary handling modes
    cval = [0.0, 0.5, 1.0]  # Constant fill values (used only for 'constant' mode)

    # Create all parameter combinations
    param_grid = list(itertools.product(filter_size, mode, cval))
    results = []

    combination_count = 0
    for filter_size, mode, cval in param_grid:
        try:
            print(f'Current combination: {combination_count + 1}/{len(param_grid)}')
            # Apply Median filtering
            median_data = apply_median_filter(original_data, filter_size, mode=mode, cval=cval)
            
            # Match dimensions if necessary (in case of boundary effects)
            median_data_matched = match_dimensions(original_data, median_data)
            
            # Calculate SSIM
            ssim_val = calculate_ssim(original_data, median_data_matched)

            # Add result to the list of results
            results.append({
                'filter': filter_size,
                'mode': mode,
                'cval': cval,
                'ssim': ssim_val
            })
            combination_count += 1
        except Exception as e:
            print(f"Error with combination (filter={filter_size}, mode={mode}, cval={cval}): {str(e)}")
            continue

    # Find the best combination
    best_result = max(results, key=lambda x: x['ssim'])
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
    # Run grid search for Median filtering
    best_params, all_results = grid_search_median_ssim(file_path)

    # Save results
    output_file_name = "median_ssim_results.csv"
    best_output_file_name = "median_ssim_best.csv"
    save_results(best_params, all_results, output_file_name, best_output_file_name)

    print("Best Parameters:")
    print(best_params)

    print("\nAll Results:")
    for result in all_results:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})