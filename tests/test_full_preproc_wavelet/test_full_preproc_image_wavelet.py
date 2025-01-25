import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..')))

from src.utils.preprocess import load_nii, get_data, resample_image, normalize_data
from src.utils.preprocess import match_dimensions

from src.utils.preprocess import extract_brain, crop_to_largest_bounding_box, apply_wavelet_denoising

from src.utils.preprocess_validation import calculate_snr_with_mask
from src.utils.preprocess_validation import calculate_mse, calculate_psnr
from src.utils.preprocess_validation import calculate_cnr, calculate_ssim
from src.utils.preprocess_validation import calculate_relative_psnr, calculate_relative_rmse
#from src.utils.preprocess_validation import plot_slices

def evaluate_raw_images(paths_to_images):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results = []

    for path in paths_to_images:
        file_path = os.path.join(current_dir, '..', '..', path)

        nii = load_nii(file_path)
        original_data = get_data(nii)

        resampled_image = resample_image(nii)
        normalized_data = normalize_data(resampled_image)

        brain_data = extract_brain(normalized_data, what_to_return={'extracted_brain': 'numpy', 'mask': 'numpy'})
        extracted_brain = brain_data['extracted_brain']
        mask = brain_data['mask']
        cropped_data = crop_to_largest_bounding_box(extracted_brain, mask=mask)
        smoothed_data = apply_wavelet_denoising(cropped_data)
        smoothed_normalized_data = normalize_data(smoothed_data)

        # Compute metrics
        snr_value = calculate_snr_with_mask(smoothed_normalized_data)
        cnr_value = calculate_cnr(smoothed_normalized_data)

        smoothed_data_matched = match_dimensions(original_data, smoothed_normalized_data)

        mse_value = calculate_mse(original_data, smoothed_data_matched)
        psnr_value = calculate_psnr(original_data, smoothed_data_matched, mse=mse_value)
        ssim_value = calculate_ssim(original_data, smoothed_data_matched)

        relative_psnr_value = calculate_relative_psnr(smoothed_data)
        relative_rmse_value = calculate_relative_rmse(smoothed_data)

        results.append({'path': path, 
                        'snr': snr_value, 
                        'cnr': cnr_value,
                        'mse': mse_value,
                        'psnr': psnr_value, 
                        'ssim': ssim_value,
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
    ax.imshow(original_data[:, :, 130], cmap="gray", alpha=1)
    ax.axis("off")
    #ax.set_title(title)
    if isSave_slices:
        save_fig_to = 'best_image_slices_full_preproc_wavelet.png'
        plt.savefig(save_fig_to)

    print(f"Selected slices saved to {save_fig_to}")

if __name__ == "__main__":
    # Get the current working directory of the script 
    rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"

    paths_to_images = [rel_file_path]
    results, best_image = evaluate_raw_images(paths_to_images)

    path_to_all_results = 'test_full_preproc_image_wavelet_all_results.csv'
    path_to_best_image = 'test_full_preproc_image_wavelet_best_image.csv'

    save_results(results, best_image, path_to_all_results, path_to_best_image)

    plot_save_selected_slices(path_to_best_image, isSave_slices=True)
