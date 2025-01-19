import os
import sys
import itertools
import pandas as pd

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.preprocess import load_nii, resample_image
from src.utils.preprocess_validation import calculate_snr

def grid_search_resample_snr(file_path):
    """
    Perform a grid search to find the best combination of resample_image parameters 
    that yield the highest Signal-to-Noise Ratio (SNR).

    Args:
        file_path (str): Path to the NIfTI file.

    Returns:
        dict: Best parameter combination and corresponding SNR.
    """
    # Load the NIfTI file
    nii = load_nii(file_path)

    # Define parameter grid
    orders = range(0, 6)  # Spline interpolation orders
    modes = ['constant', 'nearest', 'reflect', 'wrap']  # Boundary handling modes
    cvals = [0.0, 0.5, 1.0]  # Constant fill values
    voxel_size = (1, 1, 1)  # Fixed for this example

    # Create all parameter combinations
    param_grid = list(itertools.product(orders, modes, cvals))
    results = []

    # Iterate through parameter combinations
    for order, mode, cval in param_grid:
        try:
            # Resample the image
            resampled_data = resample_image(nii, voxel_size=voxel_size, order=order, mode=mode, cval=cval, output_format='numpy')

            # Calculate SNR
            snr = calculate_snr(resampled_data)
            results.append({'order': order, 'mode': mode, 'cval': cval, 'snr': snr})
        except Exception as e:
            # Handle any errors during resampling or SNR calculation
            print(f"Error with combination (order={order}, mode={mode}, cval={cval}): {str(e)}")
            continue

    # Find the best combination
    best_result = max(results, key=lambda x: x['snr'])
    return best_result, results

def save_results(best_params, 
                 all_results, 
                 output_file_name: str, 
                 best_output_file_name: str):
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
    best_params, all_results = grid_search_resample_snr(file_path)

    output_file_name = "resample_image_snr.csv"
    best_output_file_name = "resample_image_snr_best.csv"

    save_results(best_params, all_results, output_file_name, best_output_file_name)

    print("Best Parameters:")
    print(best_params)

    print("\nAll Results:")
    for result in all_results:
        print(result)
