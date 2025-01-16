
# TODO: deal with cropping and smoothing

import os

import torch

# For smoothing, wavelet denoising
import pywt

import ants
# For DL brain extraction
import antspynet

import numpy as np

from typing import Union

# For loading nifti files
import nibabel as nib

# For smoothing, nibabel smooth
from nibabel.processing import smooth_image
from nibabel.processing import resample_to_output 

# For cropping and resampling
from nilearn.image import crop_img #, resample_img 

from scipy.ndimage import gaussian_filter # for smoothing, gaussian filter

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

def resample_image(nii: nib.Nifti1Image, 
                   voxel_size: tuple=(1, 1, 1),
                   output_format: str='numpy') -> Union[nib.Nifti1Image, np.ndarray]:
    """
    Resample a nifti image to a given voxel size
    
    Args:
    nii: nib.Nifti1Image: nifti image object
    voxel_size: tuple: voxel size for resampling
    output_format: str: output format, either 'nib' or 'numpy'
    
    Returns:
    Union[nib.Nifti1Image, np.ndarray]: resampled nifti image object or numpy array

    Raises:
    TypeError: If input types are incorrect
    ValueError: If voxel_size or output_format are invalid
    RuntimeError: If resampling fails
    """
    if not isinstance(nii, nib.Nifti1Image): # Ensure the image is a nibabel Nifti1Image
        raise TypeError(f"Expected nib.Nifti1Image, got {type(nii)}")

    if not isinstance(voxel_size, tuple) or len(voxel_size) != 3:
        raise ValueError(f"voxel_size must be a tuple of length 3, got {voxel_size}")

    if not all(isinstance(x, (int, float)) and x > 0 for x in voxel_size):
        raise ValueError(f"All elements in voxel_size must be positive numbers, got {voxel_size}")

    if not isinstance(output_format, str):
        raise TypeError(f"output_format must be a string, got {type(output_format)}")

    try:
        resampled = resample_to_output(nii, voxel_sizes=voxel_size)
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
                   eps: float = 1e-8) -> np.ndarray:
    """
    Normalize the data using z-score or min-max method
    f
    Args:
    data: np.ndarray: data to be normalized
    method: str: normalization method, either "z-score" or "min-max"
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
            min_value = np.min(data)
            delta_value = np.max(data) - min_value
            if delta_value == 0:
                raise ZeroDivisionError("No variation in the data. Cannot perform min-max normalization")
            return (data - min_value) / (delta_value + eps) # TODO: deprecate eps?
        
        else:
            raise ValueError("Invalid normalization method. Choose 'z-score' or 'min-max'.")

    except Exception as e:
        raise RuntimeError(f"An error occurred while normalizing the data: {str(e)}")
    
def extract_brain(data: np.ndarray, 
                  verbose: bool = True) -> np.ndarray:
    """
    Extract brain from a given image using deep learning brain extraction
    
    Args:
    data: np.ndarray: image data
    verbose: bool: whether to print the output
    
    Returns:
    np.ndarray: brain extracted image data

    Raises: TypeError: If input data is not a numpy array 
    RuntimeError: If brain extraction fails
    """
    # TODO: test other modalities? ['t1', 't1nobrainer', 't1combined']
    # t1 - ANTs-flavored
    # t1nobrainer - FreeSurfer-flavored
    # t1combined - combined

    if not isinstance(data, np.ndarray): # Ensure the data is a numpy array
        raise TypeError(f"Expected numpy array, got {type(data)}")
    
    try:
        image = ants.from_numpy(data)

        if image is None:
            raise RuntimeError("Failed to initialize ants.Image object")
    
        # Perform brain extraction # Using 't1' modality for brain extraction
        mask = antspynet.brain_extraction(image, modality='t1', verbose=verbose)

        if mask is None:
            raise RuntimeError("Failed to perform brain extraction")
        
        # Apply mask to the image
        brain = image * mask
        return brain.numpy()
    
    except ants.AntsrError as e:
        raise RuntimeError(f"An error occurred while performing brain extraction: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")

# TODO: fix cropping, choose the most optimal
def crop_nilearn(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Crop the image to remove zero values
    
    Args:
    nii: nib.Nifti1Image: nifti image object
    verbose: bool: whether to print the output
    
    Returns:
    nib.Nifti1Image: cropped nifti image object
    """
    cropped = crop_img(nii, copy=True, rtol=1e-8, copy_header=True)
    return cropped

def crop_numpy(data, target_shape = (192, 192, 192)):
    current_shape = data.shape
    padding = [(max((t - c) // 2, 0), max((t - c + 1) // 2, 0)) for t, c in zip(target_shape, current_shape)]
    cropped_or_padded = np.pad(data, padding, mode='constant')
    slices = tuple(slice(max((c - t) // 2, 0), max((c - t) // 2, 0) + t) for c, t in zip(current_shape, target_shape))
    return cropped_or_padded[slices]

# TODO: test different smoothing methods, choose the most optimal?
def smooth_gaussian(data: np.ndarray, 
                    sigma: float = 1.0) -> np.ndarray:
    """
    Smooth the image using gaussian filter
    
    Args:
    data: np.ndarray: image data
    sigma: float: standard deviation for gaussian filter

    Returns:
    np.ndarray: smoothed image data
    """
    return gaussian_filter(data, sigma=sigma)

def smooth_nibabel(nii: nib.Nifti1Image, 
                   sigma: float = 1.0) -> nib.Nifti1Image:
    """
    Smooth the image using nibabel smooth
    
    Args:
    nii: nib.Nifti1Image: nifti image object
    sigma: float: standard deviation for gaussian filter
    
    Returns:
    nib.Nifti1Image: smoothed nifti image object
    """
    smoothed = smooth_image(nii, sigma=sigma)
    return smoothed

def smooth_wavelet_denoise(data: np.ndarray, 
                           wavelet: str = 'db1', 
                           mode: str = 'soft', 
                           threshold=None) -> np.ndarray:
    """
    Smooth the image using wavelet denoising
    
    Args:
    data: np.ndarray: image data
    wavelet: str: wavelet type
    mode: str: thresholding mode
    threshold: float: threshold value
    
    Returns:
    np.ndarray: smoothed image data
    """
    coeffs = pywt.wavedecn(data, wavelet=wavelet)
    threshold = threshold or (0.1 * np.max(data))
    denoised_coeffs = pywt.threshold(coeffs, threshold, mode=mode)
    return pywt.waverecn(denoised_coeffs, wavelet=wavelet)

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

def preprocess_image(image: nib.Nifti1Image) -> np.ndarray:
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
        cropped = crop_numpy(extracted) # TODO: fix cropping
    except Exception as e:
        raise RuntimeError(f"Cropping failed: {str(e)}")

    try:
        # Smoothing
        smoothed = smooth_gaussian(cropped) # TODO: fix smoothing
    except Exception as e:
        raise RuntimeError(f"Smoothing failed: {str(e)}")
    
    return smoothed
