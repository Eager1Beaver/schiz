import os
import torch
from logger import logger
from utils.preprocess import (
    normalize_data,
    extract_brain,
    crop_to_largest_bounding_box,
    apply_gaussian_smoothing,
)

def preprocess_images(
    raw_nii_dir: str = "data/raw_nii",
    train_set_dir: str = "train_set",
    test_set_dir: str = "test_set",
    is_normalize: bool = False,
    norm_method: str = "min-max",
    min_max_min_val: float = 0,
    min_max_max_val: float = 1,
    is_brain_extraction: bool = True, 
    brain_extraction_modality: str = "t1",
    brain_extraction_verbose: bool = False,
    is_crop: bool = False, 
    is_smooth: bool = False,
    smooth_sigma: float = 1.5,
    smooth_order: int = 2,
    smooth_mode: str = "constant",
    smooth_cval: int = 1,
    smooth_truncate: float = 2, 
    is_re_normalize_after_smooth: bool = False, 
    is_preprocess_test_set: bool = False,
    output_dir: str = "preprocessed",
) -> None:
    """
    Preprocesses MRI images based on user-defined steps.
    
    Supports: Normalization, Brain Extraction, Cropping, Smoothing, and Re-Normalization.

    Parameters:
        raw_nii_dir (str): Directory containing raw .nii images.
        train_set_dir (str): Directory for training set.
        test_set_dir (str): Directory for test set.
        is_normalize (bool): Apply normalization.
        norm_method (str): Normalization method.
        min_max_min_val (float): Min value for Min-Max normalization.
        min_max_max_val (float): Max value for Min-Max normalization.
        is_brain_extraction (bool): Apply brain extraction.
        brain_extraction_modality (str): Modality for extraction.
        brain_extraction_verbose (bool): Verbose output for brain extraction.
        is_crop (bool): Crop using brain mask (if available).
        is_smooth (bool): Apply Gaussian smoothing.
        smooth_sigma (float): Gaussian smoothing sigma.
        smooth_order (int): Order of the derivative filter.
        smooth_mode (str): Mode for smoothing.
        smooth_cval (int): Constant value for out-of-bounds pixels.
        smooth_truncate (float): Truncate smoothing filter.
        is_re_normalize_after_smooth (bool): Re-normalize after smoothing.
        is_preprocess_test_set (bool): Apply preprocessing to test set.
        output_dir (str): Directory for preprocessed images.

    Returns:
        None
    """
    datasets = [train_set_dir, test_set_dir] if is_preprocess_test_set else [train_set_dir]

    for dataset in datasets:
        input_dir = os.path.join(raw_nii_dir, dataset)
        save_dir = os.path.join(input_dir, output_dir)
        os.makedirs(save_dir, exist_ok=True)

        for img_name in os.listdir(input_dir):
            if not img_name.endswith(".pt"): 
                continue
            
            img_path = os.path.join(input_dir, img_name)
            data = torch.load(img_path).numpy()

            # Step 1: Normalize (Before Brain Extraction)
            if is_normalize:
                data = normalize_data(
                    data, method=norm_method, min_val=min_max_min_val, max_val=min_max_max_val
                )

            # Test set are only resampled and normalized
            if dataset == train_set_dir:
                # Step 2: Brain Extraction
                mask_data = None
                if is_brain_extraction:
                    extracted = extract_brain(
                        data, 
                        modality=brain_extraction_modality, 
                        what_to_return={"extracted_brain": "numpy", "mask": "numpy"},
                        verbose=brain_extraction_verbose,
                    )
                    data, mask_data = extracted["extracted_brain"], extracted["mask"]

                # Step 3: Cropping (Uses mask if available, else whole image)
                if is_crop:
                    if mask_data is not None:
                        data = crop_to_largest_bounding_box(data=data, mask=mask_data)
                    else:
                        logger.warning(f"Warning: Cropping enabled but no mask found for {img_name}. Skipping crop.")

                # Step 4: Gaussian Smoothing
                if is_smooth:
                    data = apply_gaussian_smoothing(
                        data,
                        sigma=smooth_sigma,
                        order=smooth_order,
                        mode=smooth_mode,
                        cval=smooth_cval,
                        truncate=smooth_truncate,
                    )

                    # Optional: Re-normalize after smoothing
                    if is_re_normalize_after_smooth:
                        data = normalize_data(data, method=norm_method, min_val=min_max_min_val, max_val=min_max_max_val)
            else:
                pass

            # Save the processed image
            torch.save(torch.tensor(data), os.path.join(save_dir, img_name))
