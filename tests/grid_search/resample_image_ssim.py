import os
import sys
import itertools
import numpy as np
import pandas as pd

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.preprocess import load_nii, get_data, resample_image
from src.preprocess_validation import calculate_ssim

def match_dimensions(original_image, modified_image):
    """
    Pad or crop modified_image to match the dimensions of original_image.
    """
    diff = np.array(original_image.shape) - np.array(modified_image.shape)
    pad = [(0, max(0, d)) for d in diff]  # Calculate padding required
    crop = [slice(0, min(s1, s2)) for s1, s2 in zip(original_image.shape, modified_image.shape)]

    # Pad if smaller, crop if larger
    matched = np.pad(modified_image, pad_width=pad, mode='constant') if np.any(diff > 0) else modified_image[tuple(crop)]
    return matched[:original_image.shape[0], :original_image.shape[1], :original_image.shape[2]]  # Ensure final shape matches exactly

def grid_search_resample_ssim(file_path):
    """
    Perform a grid search to find the best combination of resample_image parameters 
    that yield the highest Structural Similarity Index (SSIM)

    Args:
        file_path (str): Path to the NIfTI file.

    Returns:
        dict: Best parameter combination and corresponding SNR.
    """
    # Load the NIfTI file
    nii = load_nii(file_path)
    original_data = get_data(nii)  # Extract data from the NIfTI object

    # Define parameter grid (71 combinations)
    orders = range(0, 6)  # Spline interpolation orders
    modes = ['constant', 'nearest', 'reflect', 'wrap']  # Boundary handling modes
    cvals = [0.0, 0.5, 1.0]  # Constant fill values
    voxel_size = (1, 1, 1)  # Fixed for this example

    # Create all parameter combinations
    param_grid = list(itertools.product(orders, modes, cvals))
    results = []

    combination_count = 0
    # Iterate through parameter combinations
    for order, mode, cval in param_grid:
        try:
            print(f'Current combination count: {combination_count}/71')
            # Resample the image
            resampled_data = resample_image(nii, voxel_size=voxel_size, order=order, mode=mode, cval=cval, output_format='numpy')
            resampled_data_matched = match_dimensions(original_data, resampled_data)
            #print('shape orig', original_data.shape, 'shape res', resampled_data.shape)
            # Calculate SSIM
            ssim_val = calculate_ssim(original_data, resampled_data_matched)

            # Add result to the list of results
            results.append({'order': order, 'mode': mode, 'cval': cval, 'ssim': ssim_val})
            combination_count += 1
        except Exception as e:
            # Handle any errors during resampling or SNR calculation
            print(f"Error with combination (order={order}, mode={mode}, cval={cval}): {str(e)}")
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
    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    file_path = os.path.join(current_dir, '..', '..', rel_file_path)

    #file_path = "../../data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"

    # Run grid search
    best_params, all_results = grid_search_resample_ssim(file_path)

    output_file_name = "resample_image_ssim.csv"
    best_output_file_name = "resample_image_ssim_best.csv"

    save_results(best_params, all_results, output_file_name, best_output_file_name)

    print("Best Parameters:")
    print(best_params)

    print("\nAll Results:")
    for result in all_results:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})
       