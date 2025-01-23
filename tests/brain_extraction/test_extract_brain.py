import os
import sys
#import ants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocess import load_nii, get_data, resample_image, normalize_data, extract_brain
from src.utils.preprocess_validation import calculate_snr_with_mask, calculate_mse, calculate_psnr, calculate_ssim


"""
Results:

t1 - passes
t1nobrainer - bad_alloc
t1combined - bad_alloc

t2 - passes
"""

# TODO: ants.plot doesn't support displaying multiple subplots
# But it can be implemented using pure numpy

'''def plot_brain_extractions(images_with_masks, modalities) -> None:
    """
    Plot multiple brain extraction results in one figure.

    Args:
        images_with_masks (list of tuples): Each tuple contains an ANTsImage and its corresponding mask.
        modalities (list of str): List of modality names for titles.
    """
    """num_plots = len(images_with_masks)
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))  # Adjust figure size as needed

    for i, (image, mask) in enumerate(images_with_masks):
        ax = axes[i] if num_plots > 1 else axes  # Handle case with one plot
        ants.plot(image, overlay=mask, overlay_alpha=0.5)
        ax.set_title(modalities[i])

    plt.tight_layout()
    plt.show()"""

    #num_plots = len(images_with_masks)
    #fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))  # Adjust figure size as needed

    for i, (image, mask) in enumerate(images_with_masks):
        #ax = axes[i] if num_plots > 1 else axes  # Handle case with one plot
        ants.plot(image, overlay=mask, overlay_alpha=0.5)
        #ax.set_title(modalities[i])

    #plt.tight_layout()
    #plt.show()

def main(file_path, modalities):
    nii = load_nii(file_path)
    resampled_image = resample_image(nii)
    normalized_image = normalize_data(resampled_image)

    images_with_masks = []
    for modality in modalities:
        image, mask = extract_brain(data=normalized_image, modality=modality, return_mask=True, verbose=True)
        images_with_masks.append((image, mask))
    
    plot_brain_extractions(images_with_masks, modalities)

if __name__ == '__main__':
    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path_t1 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    rel_file_path_t2 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_T2w.nii.gz"
    file_path = os.path.join(current_dir, '..', rel_file_path_t2)

    #modalities = ['t1', 't1nobrainer', 't1combined']
    modalities = ['t2']
    
    main(file_path, modalities)'''


# Alternative implementation:
def plot_brain_extraction(image, mask, ax, title=""):
    """
    Plot the brain extraction result using Matplotlib.

    Parameters:
    - image (ANTsImage): The brain image.
    - mask (ANTsImage): The brain mask.
    - ax (matplotlib.axes.Axes): Axis to plot the result.
    - title (str): Title of the plot.
    """
    #image_data = image.numpy()
    #mask_data = mask.numpy()
    image_data = image
    mask_data = mask

    # Overlay mask onto the image
    composite = np.ma.masked_where(mask_data == 0, mask_data)

    ax.imshow(image_data[:, :, 130], cmap="gray", alpha=0.7)
    ax.imshow(composite[:, :, 130], cmap="Reds", alpha=0.5)  # Mask in red
    ax.axis("off")
    ax.set_title(title)

'''def main(file_path, modalities, preprocess, plot_extraction_result):
    nii = load_nii(file_path)

    if preprocess:
        resampled_data = resample_image(nii)
        normalized_data = normalize_data(resampled_data)
        data = normalized_data  # Replace the original data with the normalized one for further processing
    else:
        data = get_data(nii)

    fig, axes = plt.subplots(1, len(modalities), figsize=(15, 5))
    for ax, modality in zip(axes, modalities):
        result = extract_brain(data=data, 
                                    modality=modality, 
                                    what_to_return={'image':'numpy', 'mask':'numpy', 'extracted_brain':'numpy'}, 
                                    verbose=True)
        image, mask, extracted_brain = result['image'], result['mask'], result['extracted_brain']  # Extract brain image and mask from result
        if plot_extraction_result: 
            plot_brain_extraction(image, mask, ax=ax, title=f"Modality: {modality}")

    if plot_extraction_result:
        plt.tight_layout()
        plt.show()

    return normalized_image, extracted_brain'''

def main(file_path, modalities, preprocess, plot_extraction_result):
    nii = load_nii(file_path)

    if preprocess:
        resampled_data = resample_image(nii)
        normalized_data = normalize_data(resampled_data)
        data = normalized_data  # Replace the original data with the normalized one for further processing
    else:
        data = get_data(nii)

    #fig, axes = plt.subplots(1, len(modalities), figsize=(15, 5))
    #for ax, modality in zip(axes, modalities):
    for modality in modalities:
        result = extract_brain(data=data, 
                                    modality=modality, 
                                    what_to_return={'image':'numpy', 'mask':'numpy', 'extracted_brain':'numpy'}, 
                                    verbose=True)
        image, mask, extracted_brain = result['image'], result['mask'], result['extracted_brain']  # Extract brain image and mask from result
        if plot_extraction_result: 
            #plot_brain_extraction(image, mask, ax=ax, title=f"Modality: {modality}")
            pass

    if plot_extraction_result:
        plt.tight_layout()
        plt.show()

    return data, extracted_brain

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path_t1 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    rel_file_path_t2 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_T2w.nii.gz"
    file_path = os.path.join(current_dir, '..', rel_file_path_t2)    
    
    #modalities = ['t1', 't2']
    modalities = ['t2']

    do_preprocess = True
    plot_extraction_result = False

    normalized_image, extracted_brain = main(file_path, modalities, do_preprocess, plot_extraction_result)

    # Calculate metrics
    snr_value = calculate_snr_with_mask(extracted_brain)
    #cropped_slices_matched = match_dimensions(extracted_brain, cropped_slices)
    mne_value = calculate_mse(normalized_image, extracted_brain)
    psnr_value = calculate_psnr(normalized_image, extracted_brain, mse=mne_value)
    ssim_value = calculate_ssim(normalized_image, extracted_brain)

    # Printing the metrics
    print(f'SNR: {snr_value}\nMNE: {mne_value}\nPSNR: {psnr_value}\nSSIM: {ssim_value}\n')

    # Saving results to CSV file
    metrics_df = pd.DataFrame(
        {
            "SNR": [snr_value],
            "MSE": [mne_value],
            "PSNR": [psnr_value],
            "SSIM": [ssim_value]
        }
    )

    if do_preprocess:
        mode = 'processed'
    else:
        mode = 'clean'

    path_to_save_metrics = f"extract_brain_metrics_{mode}_t2.csv"   
    metrics_df.to_csv(os.path.join(current_dir, path_to_save_metrics), index=False)

    print(f'Metrics are saved to: {path_to_save_metrics}')

   