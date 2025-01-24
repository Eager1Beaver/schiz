import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.utils.preprocess import load_nii, get_data, match_dimensions 
from src.utils.preprocess import resample_image, normalize_data
from src.utils.preprocess import extract_brain
from src.utils.preprocess import get_largest_brain_mask_slice, crop_to_largest_bounding_box
from src.utils.preprocess import apply_gaussian_smoothing

from src.utils.preprocess_validation import plot_slices, calculate_mse
from src.utils.preprocess_validation import calculate_snr_with_mask, calculate_psnr
from src.utils.preprocess_validation import calculate_mse, calculate_ssim

from src.utils.augmentation import AddGaussianNoise

from torchvision import transforms


def augment_data(preprocessed_data, mean, std, spatial, alpha, range):

    augmentations = transforms.Compose([AddGaussianNoise(mean=mean, std=std)])

    return augmentations


def complete_preprocess(file_path: str):

    loaded_nii = load_nii(file_path)
    resampled_data = resample_image(loaded_nii)
    normalized_data = normalize_data(resampled_data)
    extracted_data = extract_brain(normalized_data)
    largest_slice = get_largest_brain_mask_slice(extracted_data)
    cropped_data = crop_to_largest_bounding_box(largest_slice)
    smoothed_data = apply_gaussian_smoothing(cropped_data)

    # Define parameter grid
    means = [0]
    stds = [0.005, 0.01, 0.02, 0.05]
    spatial_variation = [None, "radial"]
    intensity_dependent_alpha = [None, 0.05]
    dynamic_range = ["normalized"]

    # Create all parameter combinations
    param_grid = list(itertools.product(means, stds, spatial_variation, intensity_dependent_alpha, dynamic_range))
    results = []

    combination_count = 0
    for mean, std, spatial, alpha, range in param_grid:
        try:
            print(f'Current combination: {combination_count + 1}/{len(param_grid)}')
            # Apply Gaussian noise
            augmented_data = augment_data(smoothed_data, mean=mean, std=std, spatial=spatial, alpha=alpha, range=range)

            # Calculate SNR
            snr_val = calculate_snr_with_mask(augmented_data)

            # Add result to the list of results
            results.append({
                'mean': mean,
                'std': std,
                'spatial variation': spatial,
                'intensity dependent alpha': alpha,
                'dynamic range': range,
                'snr': snr_val
            })
            combination_count += 1
        except Exception as e:
            print(f"Error with combination (mean={mean}, std={std}, spatial variation={spatial}, intensity dependent alpha={alpha}, dynamic range={range}): {str(e)}")
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
    # Run grid search for Gaussian noise
    best_params, all_results = complete_preprocess(file_path)

    # Save results
    output_file_name = "gaussian_noise_snr_results.csv"
    best_output_file_name = "gaussian_noise_snr_best.csv"
    save_results(best_params, all_results, output_file_name, best_output_file_name)

    print("Best Parameters:")
    print(best_params)

    print("\nAll Results:")
    for result in all_results:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})
