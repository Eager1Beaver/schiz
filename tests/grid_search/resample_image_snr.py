import os
import sys
import itertools
import pandas as pd

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.preprocess import load_nii, resample_image
from src.preprocess_validation import calculate_snr

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

if __name__ == "__main__":
    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    file_path = os.path.join(current_dir, '..', '..', rel_file_path)

    #file_path = "../../data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"

    # Run grid search
    best_params, all_results = grid_search_resample_snr(file_path)

    # Convert all_results (list of dictionaries) into a DataFrame
    all_results_df = pd.DataFrame(all_results)

    # Save all results to CSV
    output_file_name = "resample_image_snr.csv"
    all_results_df.to_csv(output_file_name, index=False)

    # Save best_params separately
    best_params_df = pd.DataFrame([best_params])
    best_output_file_name = "resample_image_snr_best.csv"
    best_params_df.to_csv(best_output_file_name, index=False)

    print(f"Results saved to {output_file_name}")
    print(f'Best params saved to {best_output_file_name}')

    print("Best Parameters:")
    print(best_params)

    print("\nAll Results:")
    for result in all_results:
        print(result)

#
# Terminal output:
'''
Best Parameters:
{'order': 1, 'mode': 'wrap', 'cval': 0.0, 'snr': 0.6976139860383316}

All Results:
{'order': 0, 'mode': 'constant', 'cval': 0.0, 'snr': 0.6689048859136598}
{'order': 0, 'mode': 'constant', 'cval': 0.5, 'snr': 0.6709288233244372}
{'order': 0, 'mode': 'constant', 'cval': 1.0, 'snr': 0.6709288233244372}
{'order': 0, 'mode': 'nearest', 'cval': 0.0, 'snr': 0.6866165802266236}
{'order': 0, 'mode': 'nearest', 'cval': 0.5, 'snr': 0.6866165802266236}
{'order': 0, 'mode': 'nearest', 'cval': 1.0, 'snr': 0.6866165802266236}
{'order': 0, 'mode': 'reflect', 'cval': 0.0, 'snr': 0.688676129866993}
{'order': 0, 'mode': 'reflect', 'cval': 0.5, 'snr': 0.688676129866993}
{'order': 0, 'mode': 'reflect', 'cval': 1.0, 'snr': 0.688676129866993}
{'order': 0, 'mode': 'wrap', 'cval': 0.0, 'snr': 0.6899592703007589}
{'order': 0, 'mode': 'wrap', 'cval': 0.5, 'snr': 0.6899592703007589}
{'order': 0, 'mode': 'wrap', 'cval': 1.0, 'snr': 0.6899592703007589}
{'order': 1, 'mode': 'constant', 'cval': 0.0, 'snr': 0.6761229103147203}
{'order': 1, 'mode': 'constant', 'cval': 0.5, 'snr': 0.678182488829577}
{'order': 1, 'mode': 'constant', 'cval': 1.0, 'snr': 0.678182488829577}
{'order': 1, 'mode': 'nearest', 'cval': 0.0, 'snr': 0.694131269522447}
{'order': 1, 'mode': 'nearest', 'cval': 0.5, 'snr': 0.694131269522447}
{'order': 1, 'mode': 'nearest', 'cval': 1.0, 'snr': 0.694131269522447}
{'order': 1, 'mode': 'reflect', 'cval': 0.0, 'snr': 0.6962770277544591}
{'order': 1, 'mode': 'reflect', 'cval': 0.5, 'snr': 0.6962770277544591}
{'order': 1, 'mode': 'reflect', 'cval': 1.0, 'snr': 0.6962770277544591}
{'order': 1, 'mode': 'wrap', 'cval': 0.0, 'snr': 0.6976139860383316}
{'order': 1, 'mode': 'wrap', 'cval': 0.5, 'snr': 0.6976139860383316}
{'order': 1, 'mode': 'wrap', 'cval': 1.0, 'snr': 0.6976139860383316}
{'order': 2, 'mode': 'constant', 'cval': 0.0, 'snr': 0.670503745433568}
{'order': 2, 'mode': 'constant', 'cval': 0.5, 'snr': 0.6725356225640768}
{'order': 2, 'mode': 'constant', 'cval': 1.0, 'snr': 0.6725356225640768}
{'order': 2, 'mode': 'nearest', 'cval': 0.0, 'snr': 0.6882835508166814}
{'order': 2, 'mode': 'nearest', 'cval': 0.5, 'snr': 0.6882835508166814}
{'order': 2, 'mode': 'nearest', 'cval': 1.0, 'snr': 0.6882835508166814}
{'order': 2, 'mode': 'reflect', 'cval': 0.0, 'snr': 0.690332537956387}
{'order': 2, 'mode': 'reflect', 'cval': 0.5, 'snr': 0.690332537956387}
{'order': 2, 'mode': 'reflect', 'cval': 1.0, 'snr': 0.690332537956387}
{'order': 2, 'mode': 'wrap', 'cval': 0.0, 'snr': 0.6916531503113972}
{'order': 2, 'mode': 'wrap', 'cval': 0.5, 'snr': 0.6916531503113972}
{'order': 2, 'mode': 'wrap', 'cval': 1.0, 'snr': 0.6916531503113972}
{'order': 3, 'mode': 'constant', 'cval': 0.0, 'snr': 0.6702293479492845}
{'order': 3, 'mode': 'constant', 'cval': 0.5, 'snr': 0.6722598751058503}
{'order': 3, 'mode': 'constant', 'cval': 1.0, 'snr': 0.6722598751058503}
{'order': 3, 'mode': 'nearest', 'cval': 0.0, 'snr': 0.6879933605504754}
{'order': 3, 'mode': 'nearest', 'cval': 0.5, 'snr': 0.6879933605504754}
{'order': 3, 'mode': 'nearest', 'cval': 1.0, 'snr': 0.6879933605504754}
{'order': 3, 'mode': 'reflect', 'cval': 0.0, 'snr': 0.690037772771273}
{'order': 3, 'mode': 'reflect', 'cval': 0.5, 'snr': 0.690037772771273}
{'order': 3, 'mode': 'reflect', 'cval': 1.0, 'snr': 0.690037772771273}
{'order': 3, 'mode': 'wrap', 'cval': 0.0, 'snr': 0.6913621441623936}
{'order': 3, 'mode': 'wrap', 'cval': 0.5, 'snr': 0.6913621441623936}
{'order': 3, 'mode': 'wrap', 'cval': 1.0, 'snr': 0.6913621441623936}
{'order': 4, 'mode': 'constant', 'cval': 0.0, 'snr': 0.6697917792797825}
{'order': 4, 'mode': 'constant', 'cval': 0.5, 'snr': 0.6718201608468763}
{'order': 4, 'mode': 'constant', 'cval': 1.0, 'snr': 0.6718201608468763}
{'order': 4, 'mode': 'nearest', 'cval': 0.0, 'snr': 0.6875315673343239}
{'order': 4, 'mode': 'nearest', 'cval': 0.5, 'snr': 0.6875315673343239}
{'order': 4, 'mode': 'nearest', 'cval': 1.0, 'snr': 0.6875315673343239}
{'order': 4, 'mode': 'reflect', 'cval': 0.0, 'snr': 0.689573370864185}
{'order': 4, 'mode': 'reflect', 'cval': 0.5, 'snr': 0.689573370864185}
{'order': 4, 'mode': 'reflect', 'cval': 1.0, 'snr': 0.689573370864185}
{'order': 4, 'mode': 'wrap', 'cval': 0.0, 'snr': 0.6908957122528141}
{'order': 4, 'mode': 'wrap', 'cval': 0.5, 'snr': 0.6908957122528141}
{'order': 4, 'mode': 'wrap', 'cval': 1.0, 'snr': 0.6908957122528141}
{'order': 5, 'mode': 'constant', 'cval': 0.0, 'snr': 0.6696226151307437}
{'order': 5, 'mode': 'constant', 'cval': 0.5, 'snr': 0.6716501716990279}
{'order': 5, 'mode': 'constant', 'cval': 1.0, 'snr': 0.6716501716990279}
{'order': 5, 'mode': 'nearest', 'cval': 0.0, 'snr': 0.6873542954897218}
{'order': 5, 'mode': 'nearest', 'cval': 0.5, 'snr': 0.6873542954897218}
{'order': 5, 'mode': 'nearest', 'cval': 1.0, 'snr': 0.6873542954897218}
{'order': 5, 'mode': 'reflect', 'cval': 0.0, 'snr': 0.6893945356938649}
{'order': 5, 'mode': 'reflect', 'cval': 0.5, 'snr': 0.6893945356938649}
{'order': 5, 'mode': 'reflect', 'cval': 1.0, 'snr': 0.6893945356938649}
{'order': 5, 'mode': 'wrap', 'cval': 0.0, 'snr': 0.6907150353530551}
{'order': 5, 'mode': 'wrap', 'cval': 0.5, 'snr': 0.6907150353530551}
{'order': 5, 'mode': 'wrap', 'cval': 1.0, 'snr': 0.6907150353530551} 
'''       