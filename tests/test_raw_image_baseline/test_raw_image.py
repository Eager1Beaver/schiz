import os
import sys
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..')))

from src.utils.preprocess import load_nii, get_data, apply_gaussian_smoothing
from src.utils.preprocess_validation import calculate_snr_with_mask, generate_signal_mask
from src.utils.preprocess_validation import calculate_relative_psnr
from src.utils.preprocess_validation import calculate_cnr, calculate_relative_rmse
#from src.utils.preprocess_validation import plot_slices

def evaluate_raw_images(paths_to_images):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results = []

    for path in paths_to_images:
        file_path = os.path.join(current_dir, '..', '..', path)

        nii = load_nii(file_path)
        print(f'Scan shape: {nii.shape}')
        original_data = get_data(nii)
        print('original after loading ', original_data.max(), original_data.min(), original_data.mean(), original_data.std(), original_data.shape)
        smoothed_data = apply_gaussian_smoothing(original_data, sigma=0.5, order=0, mode='reflect', cval=0, truncate=2)
        print('smoothed after loading ', smoothed_data.max(), smoothed_data.min(), smoothed_data.mean(), smoothed_data.std(), smoothed_data.shape)
        plt.imshow(smoothed_data[:, :, 140], cmap="gray", alpha=1)
        plt.show()
        # Compute metrics
        snr_value = calculate_snr_with_mask(original_data)
        signal_mask = generate_signal_mask(original_data)
        cnr_value = calculate_cnr(original_data, signal_mask)
        relative_psnr_value = calculate_relative_psnr(original_data)
        relative_rmse_value = calculate_relative_rmse(original_data)

        results.append({'path': path, 
                        'snr': snr_value, 
                        'cnr': cnr_value, 
                        'relative_psnr': relative_psnr_value, 
                        'relative_rmse': relative_rmse_value})

       
    # Find the best combination
    best_image = max(results, key=lambda x: x['snr'])

    return results, best_image

def save_results(all_results, best_image, path_to_all_results: str, path_to_best_image: str):
    """
    Save all results and exclusions to CSV files.
    """
    # Save all results
    pd.DataFrame(all_results).to_csv(path_to_all_results, index=False)

    # Save best
    pd.DataFrame([best_image]).to_csv(path_to_best_image, index=False)

    print(f"All results saved to {path_to_all_results}")
    print(f"Best image results saved to {path_to_best_image}")

def plot_save_selected_slices(path_to_best_image, isSave_slices=False):
    # Load the best image
    best_image = pd.read_csv(path_to_best_image)

    # Load the image
    nii = load_nii(best_image['path'].values[0])
    original_data = get_data(nii)

    # Plot the slices
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(original_data[:, :, 140], cmap="gray", alpha=1)
    #ax.imshow(original_data[100, :, :], cmap="gray", alpha=1)

    ax.axis("off")
    #ax.set_title(title)
    if isSave_slices:
        save_fig_to = 'best_image_slices_z140.png'
        plt.savefig(save_fig_to)

    print(f"Selected slices saved to {save_fig_to}")

if __name__ == "__main__":
    # Get the current working directory of the script 
    #rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    #rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000368/ses-20110101/anat/sub-A00000368_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    
    rel_file_path = 'data/schizconnect_COBRE_images_22613/COBRE/sub-A00014839/ses-20090101/anat/sub-A00014839_ses-20090101_acq-mprage_run-02_T1w.nii.gz'
    paths_to_images = [rel_file_path]
    results, best_image = evaluate_raw_images(paths_to_images)

    path_to_all_results = 'test_raw_image_all_results.csv'
    path_to_best_image = 'test_raw_image_best_image.csv'

    save_results(results, best_image, path_to_all_results, path_to_best_image)

    plot_save_selected_slices(path_to_best_image, isSave_slices=True)
