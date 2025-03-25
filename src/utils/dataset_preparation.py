import os
import random
import torch
from logger import logger
from utils.preprocess import load_nii, resample_image, normalize_data


def prepare_dataset(train_ratio: float = 0.8,
                    raw_nii_dir: str = "data/raw_nii",
                    train_set_output_dir: str = "train_set",
                    test_set_output_dir: str = "test_set",
                    resampling_voxel_size: tuple = (2, 2, 2),
                    normalize: bool = True,
                    norm_method: str = "min-max",
                    min_max_min_val: float = 0,
                    min_max_max_val: float = 1) -> None:
    """
    Splits MRI images into training and test sets for .nii format.
    
    .nii files are preprocessed (resampled, optionally normalized) before being saved in .pt format.

    Parameters:
        train_ratio (float): Proportion of data to use for training.
        raw_nii_dir (str): Directory containing raw .nii images.
        normalize (bool): Whether to normalize the .nii images.
        norm_method (str): Normalization method ("min-max" or other supported methods).
    """
    # Define train/test directories
    train_nii_dir, test_nii_dir = os.path.join(raw_nii_dir, train_set_output_dir), os.path.join(raw_nii_dir, test_set_output_dir)

    # Create required directories
    for directory in [train_nii_dir, test_nii_dir]:
        os.makedirs(directory, exist_ok=True)

    # Get list of .nii files and shuffle
    nii_images = [f for f in os.listdir(raw_nii_dir) if f.endswith(".nii.gz")]
    random.shuffle(nii_images)

    # Split dataset
    split_idx = int(len(nii_images) * train_ratio)
    train_images, test_images = nii_images[:split_idx], nii_images[split_idx:]

    def process_and_save_nii(image_list: list, nii_dir: str,
                             save_dir: str) -> None:
        """Processes and saves .nii images as .pt while preserving the dataset split."""
        for img in image_list:
            nii_path = os.path.join(nii_dir, img)
            if not os.path.exists(nii_path):
                continue

            nii_img = load_nii(nii_path)
            processed_img = resample_image(nii_img, voxel_size=resampling_voxel_size)
            if normalize:
                processed_img = normalize_data(processed_img,
                                               method=norm_method,
                                               min_val=min_max_min_val,
                                               max_val=min_max_max_val)

            torch.save(torch.tensor(processed_img, dtype=torch.float32), os.path.join(save_dir, img + ".pt"))

    # Process .nii images
    process_and_save_nii(train_images, raw_nii_dir, train_nii_dir)
    process_and_save_nii(test_images, raw_nii_dir, test_nii_dir)

    logger.info(f"Dataset prepared: {len(train_images)} images in train_set, {len(test_images)} images in test_set.")
