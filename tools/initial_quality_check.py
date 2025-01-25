import os
import sys
import glob
import numpy as np
import pandas as pd

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocess import load_nii, get_data, resample_image, normalize_data
from src.utils.preprocess_validation import calculate_snr_with_mask, generate_signal_mask
from src.utils.preprocess_validation import calculate_relative_psnr
from src.utils.preprocess_validation import calculate_cnr, calculate_relative_rmse

# Constants for thresholds
SNR_THRESHOLD = 5
PSNR_THRESHOLD = 15

def preprocess_scan(file_to_process):
    resampled_data = resample_image(file_to_process)
    normalized_data = normalize_data(resampled_data)
    return normalized_data

def main(path_to_data: str, limit_files: int = None):
    """
    Load all .nii.gz files from the provided path, calculate their SNR and Relative PSNR, 
    and store the results in a CSV file. Exclude files below quality thresholds.
    
    Args:
    path_to_data (str): Path to the top-level "data" folder.
    limit_files (int, optional): Limit the number of files processed per dataset (for debugging).
    
    Returns:
    tuple: (all_results, exclusions) where both are lists of dictionaries.
    """
    all_results = []
    exclusions = []

    # Counter to check progress
    progress_counter = 0

    # Iterate over each dataset
    for dataset_name in ["COBRE", "MCICShare"]:
        dataset_full_name = f"schizconnect_{dataset_name}_images_22613"
        dataset_path = os.path.join(path_to_data, dataset_full_name, dataset_name)

        anat_files = glob.glob(os.path.join(dataset_path, "sub-*", "ses-*", "anat", "*.nii.gz"))

        for i, file_path in enumerate(anat_files):
            if limit_files and i >= limit_files:
                break
            
            try:
                loaded_nii = load_nii(file_path)
                #original_data = get_data(nii)

                # Preprocess the scan to ensure consistency
                preprocessed_data = preprocess_scan(loaded_nii)

                # Generate brain mask for CNR
                mask = generate_signal_mask(preprocessed_data)

                # Calculate SNR, CNR, RMSE, and Relative PSNR
                snr_value = calculate_snr_with_mask(preprocessed_data, mask)
                cnr_value = calculate_cnr(preprocessed_data, mask)
                rmse_value = calculate_relative_rmse(preprocessed_data, np.max(preprocessed_data))  # Reference as max intensity
                rel_psnr_value = calculate_relative_psnr(preprocessed_data)

                # Append results
                result = {'file_path': file_path, 
                          'snr': snr_value, 
                          'rel_psnr': rel_psnr_value,
                          'cnr': cnr_value, 
                          'rmse': rmse_value, 
                          }
                
                all_results.append(result)

                # Check quality thresholds
                if snr_value < SNR_THRESHOLD or rel_psnr_value < PSNR_THRESHOLD:
                    exclusions.append(result)

                print(f'Proccessed a scan number {progress_counter} out of {1890}') # 985 905
                progress_counter += 1
            except Exception as e:
                print(f"Error processing file {file_path}: {e}", file=sys.stderr)

    return all_results, exclusions

def save_results(all_results, exclusions, path_to_all_results: str, path_to_exclusions: str):
    """
    Save all results and exclusions to CSV files.
    """
    # Save all results
    pd.DataFrame(all_results).to_csv(path_to_all_results, index=False)

    # Save exclusions
    pd.DataFrame(exclusions).to_csv(path_to_exclusions, index=False)

    print(f"All results saved to {path_to_all_results}")
    print(f"Exclusions saved to {path_to_exclusions}")

if __name__ == '__main__':
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    #rel_data_path = 'data'  # Path to the data folder
    #data_path = os.path.join(current_dir, '..', rel_data_path)
    data_path = 'data'
    
    all_results_csv = 'quality_check_all_results.csv'
    exclusions_csv = 'quality_check_exclusions.csv'

    # Run the main function
    all_results, exclusions = main(data_path, limit_files=2)  # Set limit_files to None for full run

    # Save results
    save_results(all_results, exclusions, all_results_csv, exclusions_csv)

    print(f'Number or all records: {len(all_results)}, excluded: {len(exclusions)}')


'''def main(file_path):
    nii = load_nii(file_path)
    data = get_data(nii)
    max_intensity = data.max()
    return max_intensity

if __name__ == '__main__':

    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))

    #rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz" # max int 658
    #rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00004087/ses-20100101/anat/sub-A00004087_ses-20100101_acq-mprage_run-01_echo-02_T1w.nii.gz" # max int 756
    rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00015201/ses-20110101/anat/sub-A00015201_ses-20110101_acq-mprage_run-01_echo-04_T1w.nii.gz" # max int 1056
    file_path = os.path.join(current_dir, '..', rel_file_path)

    max_intensity = main(file_path)
    print(f"The maximum intensity in the image is: {max_intensity}")'''

    