import os
import sys
import ants
import numpy as np
import matplotlib.pyplot as plt

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocess import load_nii, resample_image, normalize_data, extract_brain

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

def main(file_path, modalities):
    nii = load_nii(file_path)
    resampled_image = resample_image(nii)
    normalized_image = normalize_data(resampled_image)

    fig, axes = plt.subplots(1, len(modalities), figsize=(15, 5))
    for ax, modality in zip(axes, modalities):
        result = extract_brain(data=normalized_image, 
                                    modality=modality, 
                                    what_to_return={'image':'numpy', 'mask':'numpy'}, 
                                    verbose=True)
        image, mask = result['image'], result['mask']  # Extract brain image and mask from result
        plot_brain_extraction(image, mask, ax=ax, title=f"Modality: {modality}")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path_t1 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    rel_file_path_t2 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_T2w.nii.gz"
    file_path = os.path.join(current_dir, '..', rel_file_path_t2)    
    
    modalities = ['t1', 't2']

    main(file_path, modalities)

   