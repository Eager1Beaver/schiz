import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocess import load_nii, get_data, resample_image
from src.utils.preprocess_validation import plot_slices, calculate_snr, plot_histogram

"""
Results:
Passes

Original SNR: 0.73
Resampled SNR: 0.67
voxel_size: tuple = (1, 1, 1)
"""

def main(file_path):
    nii = load_nii(file_path)
    original_data = get_data(nii)
    resampled_image = resample_image(nii)

    original_snr = calculate_snr(original_data)
    resampled_snr = calculate_snr(resampled_image)

    print(f"Original SNR: {original_snr:.2f}")
    print(f"Resampled SNR: {resampled_snr:.2f}")

    # Plot slices side by side
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))  # 2 rows, 4 columns
    plot_slices(data=original_data, title="Original Image", axes=axes[0])
    plot_slices(data=resampled_image, title="Resampled Image", axes=axes[1])
    fig.suptitle("Slice Comparison")
    plt.show()

    # Calculate histogram limits
    original_hist, _ = np.histogram(original_data.flatten(), bins=50)
    resampled_hist, _ = np.histogram(resampled_image.flatten(), bins=50)
    max_y = max(original_hist.max(), resampled_hist.max())

    # Plot histograms side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 2 columns
    plot_histogram(data=original_data, title="Original Image", ax=axes[0])
    plot_histogram(data=resampled_image, title="Resampled Image", ax=axes[1])

    # Set common y-limits
    for ax in axes:
        ax.set_ylim(0, max_y)

    fig.suptitle("Histogram Comparison")
    plt.show()

if __name__ == '__main__':
    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    file_path = os.path.join(current_dir, '..', rel_file_path)

    main(file_path)