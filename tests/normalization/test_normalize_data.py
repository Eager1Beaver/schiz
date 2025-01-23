import os
import sys
import numpy as np
import pandas as pd

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocess import load_nii, get_data, normalize_data
from src.utils.preprocess_validation import calculate_snr_with_mask, calculate_mse, calculate_psnr, calculate_ssim

def evaluate_normalization_methods(original_data: np.ndarray, 
                                   output_file_name: str):
    """
    Evaluate normalization methods (min-max and z-score) using metrics and save results to a CSV file.
    
    Args:
        original_data (np.ndarray): The original data to be normalized and compared to.
        output_file_name (str): Name of the CSV file to save results.

    Returns:
        pd.DataFrame: DataFrame containing the results.
    """
    normalization_methods = ['min-max', 'z-score']
    metrics_results = []

    for method in normalization_methods:
        try:
            # Normalize data
            normalized_data = normalize_data(original_data, method=method)
            
            # Compute metrics
            snr_value = calculate_snr_with_mask(normalized_data)
            mse_value = calculate_mse(original_data, normalized_data)
            psnr_value = calculate_psnr(original_data, normalized_data, mse=mse_value)
            ssim_value = calculate_ssim(original_data, normalized_data)

            # Append results
            metrics_results.append({
                "method": method,
                "snr": snr_value,
                "mse": mse_value,
                "psnr": psnr_value,
                "ssim": ssim_value
            })

        except Exception as e:
            print(f"Error processing method '{method}': {str(e)}")
            metrics_results.append({
                "method": method,
                "error": str(e)
            })

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(metrics_results)
    results_df.to_csv(output_file_name, index=False)
    print(f"Results saved to {output_file_name}")
    return results_df

if __name__ == "__main__":
    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    file_path = os.path.join(current_dir, '..', rel_file_path)

    # Load a sample file
    nii = load_nii(file_path)
    original_data = get_data(nii)

    # Evaluate normalization methods and save to CSV
    output_file = "normalize_data_comparison.csv"
    results_df = evaluate_normalization_methods(original_data, output_file)

    # Print results
    print(results_df)
