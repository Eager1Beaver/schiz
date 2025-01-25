import os
import torch
import pywt
import ants
import antspynet
import numpy as np
from typing import Union
import nibabel as nib
from nibabel.processing import resample_to_output 
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_closing, binary_opening
from skimage.filters import threshold_otsu

"""
# pywt --- For smoothing, wavelet denoising
# antspynet --- For DL brain extraction
# nibabel --- For loading nifti files
# """

def load_nii(file_path: str) -> nib.Nifti1Image:
    """
    Load a nifti file from a given file path

    Args:
    file_path: str: path to the nifti file

    Returns:
    nib.Nifti1Image: nifti image object

    Raises:
    TypeError: If file_path is not a string
    FileNotFoundError: If the file does not exist
    ValueError: If the file is not a valid NIfTI file
    """
    if not isinstance(file_path, str): # Ensure the file_path is a string
        raise TypeError(f"Expected a string for file_path, got {type(file_path)}")
    
    if not os.path.exists(file_path): # Ensure the file_path exists
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        nii = nib.load(file_path)
    except nib.filebasedimages.ImageFileError:
        raise ValueError(f"File is not a valid NIfTI file: {file_path}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the file: {str(e)}")

    if not isinstance(nii, nib.Nifti1Image):
        raise ValueError(f"Loaded object is not a NIfTI image: {type(nii)}")

    return nii

def get_data(nii: nib.Nifti1Image) -> np.ndarray:
    """
    Get the data froma a nifti image object
    
    Args:
    nii: nib.Nifti1Image: nifti image object
    
    Returns:
    np.ndarray: nifti image data

    Raises:
    TypeError: If the input is not a nib.Nifti1Image object
    ValueError: If the image data is empty or corrupted
    """
    if not isinstance(nii, nib.Nifti1Image): # Ensure the image is a nibabel Nifti1Image
        raise TypeError(f"Expected nib.Nifti1Image, got {type(nii)}")

    try:
        data = nii.get_fdata()
    except Exception as e:
        raise ValueError(f"Failed to get data from nifti image: {str(e)}")

    if data.size == 0:
        raise ValueError("The nifti image data is empty")
    
    if not np.isfinite(data).all():
        raise ValueError("The nifti image data contains non-finite values")

    return data

def get_affine(nii: nib.Nifti1Image) -> np.ndarray:
    """
    Get the affine matrix from a nifti image object
    
    Args:
    nii: nib.Nifti1Image: nifti image object
    
    Returns:
    np.ndarray: affine matrix

    Raises:
    TypeError: If the input is not a nib.Nifti1Image object
    ValueError: If the affine matrix is not valid
    """
    if not isinstance(nii, nib.Nifti1Image): # Ensure the image is a nibabel Nifti1Image
        raise TypeError(f"Expected nib.Nifti1Image, got {type(nii)}")

    try:
        affine = nii.affine
    except AttributeError:
        raise ValueError("The nifti image object does not have an affine attribute")

    if not isinstance(affine, np.ndarray):
        raise ValueError(f"Affine matrix is not a numpy array, got {type(affine)}")

    if affine.shape != (4, 4):
        raise ValueError(f"Affine matrix has incorrect shape. Expected (4, 4), got {affine.shape}")

    if not np.isfinite(affine).all():
        raise ValueError("The affine matrix contains non-finite values")

    return affine

def match_dimensions(original_image: np.ndarray, modified_image: np.ndarray) -> np.ndarray:
    """
    Pad or crop modified_image to match the dimensions of original_image.

    Args:
    original_image: np.ndarray: original image data
    modified_image: np.ndarray: modified image data

    Returns:
    np.ndarray: resampled and padded/cropped modified image data
    """
    diff = np.array(original_image.shape) - np.array(modified_image.shape)
    pad = [(0, max(0, d)) for d in diff]  # Calculate padding required
    crop = [slice(0, min(s1, s2)) for s1, s2 in zip(original_image.shape, modified_image.shape)]

    # Pad if smaller, crop if larger
    matched = np.pad(modified_image, pad_width=pad, mode='constant') if np.any(diff > 0) else modified_image[tuple(crop)]
    return matched[:original_image.shape[0], :original_image.shape[1], :original_image.shape[2]]  # Ensure final shape matches exactly

def resample_image(data, # Union[nib.Nifti1Image, np.ndarray]
                   voxel_size: tuple=(2, 2, 2),
                   order: int = 4,
                   mode: str='reflect',
                   cval: float=0.0,
                   output_format: str='numpy') -> Union[nib.Nifti1Image, np.ndarray]:
    """
    Resample a nifti image to a given voxel size
    
    Args:
    data: Union[nib.Nifti1Image, np.ndarray]: nifti image object or numpy array
    voxel_size: tuple: voxel size for resampling
    output_format: str: output format, either 'nib' or 'numpy'
    
    Returns:
    Union[nib.Nifti1Image, np.ndarray]: resampled nifti image object or numpy array

    Raises:
    TypeError: If input types are incorrect
    ValueError: If voxel_size or output_format are invalid
    RuntimeError: If resampling fails
    """
    #if not isinstance(data, nib.Nifti1Image) or not isinstance(data, np.ndarray): # Ensure the image is a nibabel Nifti1Image
    #    raise TypeError(f"Expected nib.Nifti1Image, got {type(data)}")

    if not isinstance(voxel_size, tuple) or len(voxel_size) != 3:
        raise ValueError(f"voxel_size must be a tuple of length 3, got {voxel_size}")

    if not all(isinstance(x, (int, float)) and x > 0 for x in voxel_size):
        raise ValueError(f"All elements in voxel_size must be positive numbers, got {voxel_size}")

    if not isinstance(output_format, str):
        raise TypeError(f"output_format must be a string, got {type(output_format)}")

    try:
        #if isinstance(data, np.ndarray):
            # Convert numpy array to nibabel Nifti1Image
        #    data = nib.Nifti1Image(data, get_affine(data))
        resampled = resample_to_output(data, voxel_sizes=voxel_size, order=order, mode=mode, cval=cval)
    except Exception as e:
        raise RuntimeError(f"Resampling failed: {str(e)}")

    if output_format.lower() == 'nib':
        return resampled
    elif output_format.lower() == 'numpy':
        try: # Not filling the cache if it is already empty
            return resampled.get_fdata(caching='unchanged')
        except Exception as e:
            raise RuntimeError(f"Failed to convert resampled image to numpy array: {str(e)}")
    else:
        raise ValueError(f"Invalid output_format: {output_format}. Should be either 'nib' or 'numpy'")
    
def normalize_data(data: np.ndarray, 
                   method: str = "min-max",
                   min_val = 0, 
                   max_val = 1,
                   eps: float = 1e-8) -> np.ndarray:
    """
    Normalize the data using z-score or min-max method 
    In case of min-max method, normalize the intensity values 
    of all scans to a consistent range (e.g., [0, 1] or [0, 255])
    
    Args:
    data: np.ndarray: data to be normalized
    method: str: normalization method, either "z-score" or "min-max"
    min_val: (float): Minimum value of the target range.
    max_val: (float): Maximum value of the target range.
    eps: float: a small constant to avoid division by zero

    Returns:
    np.ndarray: normalized data

    Raises:
    TypeError: If the input data is not a numpy array
    ValueError: If the normalization method is invalid
    ZeroDivisionError: If the data array has no variation
    """

    if not isinstance(data, np.ndarray): # Ensure the data is a numpy array
        raise TypeError(f"Expected numpy array, got {type(data)}")
    
    try:
        if method == 'z-score':
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                raise ZeroDivisionError("Standard deviation is zero. Cannot perform z-score normalization")
            return (data - mean) / std
        
        elif method =='min-max':
            min_data_value = np.min(data)
            max_data_value = np.max(data)
            delta_data_value = max_data_value - min_data_value

            if delta_data_value == 0:
                raise ZeroDivisionError("No variation in the data. Cannot perform min-max normalization")
            
            return ((data - min_data_value) / (delta_data_value + eps)) * (max_val - min_val) + min_val
        
        else:
            raise ValueError("Invalid normalization method. Choose 'z-score' or 'min-max'.")

    except Exception as e:
        raise RuntimeError(f"An error occurred while normalizing the data: {str(e)}")

def refine_brain_mask(mask, structure=None): # TODO: integrate into pipeline
    closed = binary_closing(mask, structure=structure)
    opened = binary_opening(closed, structure=structure)
    return opened

def extract_brain(data: np.ndarray,
                  modality: str = 't1',
                  what_to_return: dict = {'extracted_brain': 'numpy'},
                  verbose: bool = True) -> dict:
    """
    Extract brain from a given image using deep learning brain extraction.

    Args:
        data (np.ndarray): Input 3D image data.
        modality (str): Modality for brain extraction (default: 't1').
        what_to_return (dict): Specifies objects ('image', 'mask', 'extracted_brain') to return 
                               and their types ('numpy' or 'ants').
                               Default: {'extracted_brain': 'numpy'}.
        verbose (bool): Whether to print additional output (default: True).

    Returns:
        dict: A dictionary containing the requested objects in the specified formats.

    Raises:
        TypeError: If input data is not a numpy array.
        RuntimeError: If brain extraction fails or an invalid return type is requested.
    """
    if not isinstance(data, np.ndarray):  # Ensure the data is a numpy array
        raise TypeError(f"Expected numpy array, got {type(data)}")

    try:
        # Convert numpy array to ANTs image
        image = ants.from_numpy(data)
        if image is None:
            raise RuntimeError("Failed to initialize ants.Image object")

        # Perform brain extraction
        mask = antspynet.brain_extraction(image, modality=modality, verbose=verbose)
        if mask is None:
            raise RuntimeError("Failed to perform brain extraction")

        # Apply the mask to extract the brain
        extracted_brain = image * mask

        # Prepare results based on 'what_to_return'
        result = {}
        for key, value in what_to_return.items():
            if key == 'image':
                result['image'] = image.numpy() if value == 'numpy' else image
            elif key == 'mask':
                result['mask'] = mask.numpy() if value == 'numpy' else mask
            elif key == 'extracted_brain':
                result['extracted_brain'] = extracted_brain.numpy() if value == 'numpy' else extracted_brain
            else:
                raise ValueError(f"Invalid key '{key}' in what_to_return. Valid keys are 'image', 'mask', 'extracted_brain'.")

        return result

    except ants.AntsrError as e:
        raise RuntimeError(f"An error occurred while performing brain extraction: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")  

# Step 5: Cropping using the bounding box of the largest brain area by determining that particular slice

# Determine the slice with the largest brain area
def get_largest_brain_mask_slice(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Determine the slices with the largest brain mask dimension,
    using Otsu's thresholding for binarization.
    
    Parameters:
    - mask: numpy array, brain mask (3D)
    
    Returns:
    - largest_slice_index: int, index of the slice with the largest brain mask
    """
    # Apply Otsu's thresholding to binarize the mask
    threshold = threshold_otsu(mask)
    binary_mask = (mask > threshold).astype(np.uint8)

    largest_area = 0
    largest_slice_index = -1

    # Iterate through each slice along the z-axis
    for z in range(binary_mask.shape[2]):  # Loop through slices
        slice_mask = binary_mask[:, :, z]

        # Skip completely empty slices
        if np.any(slice_mask > 0):
            # Find the bounding box of the slice mask
            coords = np.argwhere(slice_mask > 0)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0) + 1  # Inclusive

            height = x_max - x_min
            width = y_max - y_min
            area = height * width

            # Update the largest slice index based on area
            if area > largest_area:
                largest_area = area
                largest_slice_index = z

    return binary_mask, largest_slice_index

# Crop all the slices using the dimension or bounding box of the largest brain area
def crop_to_largest_bounding_box(data: np.ndarray, 
                                 processed_mask: np.ndarray = None, 
                                 largest_slice_idx: int = None,
                                 mask: np.ndarray = None) -> np.ndarray:
    """
    Crop all slices to the bounding box of the largest brain area.
    
    Parameters:
    - data: numpy array, 3D MRI data
    - mask: numpy array, binary brain mask (3D)
    - largest_slice_idx: int, index of the slice with the largest brain mask
    
    Returns:
    - cropped_data: numpy array, cropped 3D MRI data
    """
    if processed_mask is None or largest_slice_idx is None:
        processed_mask, largest_slice_idx = get_largest_brain_mask_slice(mask)

    # Extract the mask of the slice with the largest brain area
    largest_slice_mask = processed_mask[:, :, largest_slice_idx]

    # Find the bounding box of the largest slice
    coords = np.argwhere(largest_slice_mask > 0)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0) + 1  # Inclusive

    # Crop all slices using the bounding box dimensions
    cropped_slices = data[x_min:x_max, y_min:y_max, :]

    return cropped_slices

# Step 6: Apply Gaussian Smoothing (to reduce noise and improve results) after cropping
def apply_gaussian_smoothing(data, sigma=1.5, order=2, mode='constant', cval=1.0, truncate=2.0):
    """
    Apply Gaussian smoothing to the data to reduce noise and artifacts.
    Args:
        data (numpy.ndarray): 3D MRI data.
        sigma (float): Standard deviation of the Gaussian kernel.
        
    Returns:
        smoothed_data (numpy.ndarray): Smoothed 3D MRI data.
    """
    smoothed_data = gaussian_filter(data, sigma=sigma, order=order, mode=mode, cval=cval, truncate=truncate)
    return smoothed_data

# Method 2: Median filtering for smoothing
def apply_median_filter(data, filter_size=3, mode='constant', cval=0.0):
    """
    Apply median filtering to a 3D MRI image.

    Args:
        data (numpy.ndarray): Input 3D MRI image.
        filter_size (int): Size of the neighborhood for median filtering.

    Returns:
        numpy.ndarray: Median-filtered MRI image.
    """
    return median_filter(data, size=filter_size, mode=mode, cval=cval)

# Method 3: Wavelet denoising for smoothing
def apply_wavelet_denoising(data, wavelet='coif1', level=2, threshold=0.05, 
                            thresholding='hard', mode='reflect'):
    """
    Apply wavelet denoising to a 3D MRI image with added thresholding mode and boundary mode.
    
    Args:
        data (numpy.ndarray): Input 3D MRI image.
        wavelet (str): Wavelet type (e.g., 'db1', 'sym4').
        level (int): Level of wavelet decomposition.
        threshold (float): Threshold value for soft or hard thresholding.
        thresholding (str): Type of thresholding ('soft' or 'hard').
        mode (str): Boundary handling mode (e.g., 'symmetric', 'constant').
    
    Returns:
        numpy.ndarray: Denoised 3D MRI image.
    """
    denoised_slices = []
    
    for slice_idx in range(data.shape[2]):
        slice_data = data[:, :, slice_idx]
        
        # Perform 2D wavelet decomposition
        coeffs2 = pywt.wavedec2(slice_data, wavelet, mode=mode, level=level)
        
        # Apply soft or hard thresholding to detail coefficients
        thresholded_coeffs = [
            tuple(pywt.threshold(c, threshold, mode=thresholding) for c in detail) 
            for detail in coeffs2[1:]
        ]
        
        # Reconstruct the slice with thresholded coefficients
        coeffs2[1:] = thresholded_coeffs
        denoised_slice = pywt.waverec2(coeffs2, wavelet, mode=mode)
        
        # Ensure the reconstructed slice has the same shape as the input
        denoised_slice = denoised_slice[:slice_data.shape[0], :slice_data.shape[1]]
        denoised_slices.append(denoised_slice)
    
    # Stack slices back into a 3D array
    denoised_image = np.stack(denoised_slices, axis=2)
    return denoised_image

# TODO: probably to be deprecated
# because a conversion from a numpy array to a torch tensor is handled by the data_loader
# if the data augmentation is performed (it is performed on training data),
# then a numpy array is converted to a torch tensor during data augmentation step 
# otherwise it is converted by __getitem__ of the MRIDataset class
def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor
    
    Args:
    data: np.ndarray: numpy array
    
    Returns:
    torch.Tensor: PyTorch tensor
    """
    return torch.tensor(data, dtype=torch.float32)

def preprocess_image(image: nib.Nifti1Image) -> np.ndarray: # TODO: change!
    """
    Preprocess the image using the following steps:
    - Get the data
    - Normalize the data
    - Extract
    - Crop
    - Smooth

    Args:
    image: nib.Nifti1Image: nifti image object

    Returns:
    np.ndarray: preprocessed image data

    Raises: TypeError: If the input image is not an instance of nib.Nifti1Image 
    ValueError: For any processing step failures
    """
    if not isinstance(image, nib.Nifti1Image): # Ensure the image is a nibabel Nifti1Image
        raise TypeError(f"Expected nib.Nifti1Image, got {type(image)}")
    
    #data = get_data(image) # TODO: to be deprecated
    # # loading a .nii file instead 
    # and resample_image returns a np.ndarray for further processing steps

    try:
        # Resampling
        resampled = resample_image(image)
    except Exception as e:    
        raise RuntimeError(f"Resampling failed: {str(e)}")

    try:    
        # Normalization
        normalized = normalize_data(resampled)
    except Exception as e:
        raise RuntimeError(f"Normalization failed: {str(e)}")
    
    try:
        # Brain extraction
        extracted = extract_brain(normalized)
    except Exception as e:
        raise RuntimeError(f"Brain extraction failed: {str(e)}")

    try:
        # Cropping
        cropped = crop_to_largest_bounding_box(extracted) # TODO: fix cropping
    except Exception as e:
        raise RuntimeError(f"Cropping failed: {str(e)}")

    try:
        # Smoothing
        smoothed = apply_gaussian_smoothing(cropped) # TODO: fix smoothing
    except Exception as e:
        raise RuntimeError(f"Smoothing failed: {str(e)}")
    
    return smoothed
