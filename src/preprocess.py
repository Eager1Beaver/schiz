import nibabel as nib # for loading nifti files
from nibabel.processing import resample_to_output
from nibabel.processing import smooth_image # for smoothing, nibabel smooth

from nilearn.image import crop_img #, resample_img # for cropping and resampling

from scipy.ndimage import gaussian_filter # for smoothing, gaussian filter
#from scipy.ndimage import median_filter # for smoothing, median filter

import pywt # for smoothing, wavelet denoising

import ants
import antspynet # for DL brain extraction

import numpy as np

import torch

from typing import Union

def load_nii(file_path: str) -> nib.Nifti1Image:
    """
    Load a nifti file from a given file path

    Args:
    file_path: str: path to the nifti file

    Returns:
    nib.Nifti1Image: nifti image object
    """
    nii = nib.load(file_path)
    return nii

def get_data(nii: nib.Nifti1Image) -> np.ndarray:
    """
    Get the data froma a nifti image object
    
    Args:
    nii: nib.Nifti1Image: nifti image object
    
    Returns:
    np.ndarray: nifti image data
    """
    data = nii.get_fdata()
    return data

def get_affine(nii: nib.Nifti1Image) -> np.ndarray:
    """
    Get the affine matrix from a nifti image object
    
    Args:
    nii: nib.Nifti1Image: nifti image object
    
    Returns:
    np.ndarray: affine matrix"""
    affine = nii.affine
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
    nib.Nifti1Image: resampled nifti image object
    """
    resampled = resample_to_output(nii, voxel_sizes=voxel_size)

    if output_format == 'nib':
        return resampled
    elif output_format == 'numpy':
        return resampled.get_fdata()
    else:
        raise ValueError('output_format should be either "nib" or "numpy"')
    
def normalize_data(data: np.ndarray, 
                   method: str = "min-max") -> np.ndarray:
    """
    Normalize the data using z-score or min-max method
    
    Args:
    data: np.ndarray: data to be normalized
    method: str: normalization method, either "z-score" or "min-max"

    Returns:
    np.ndarray: normalized data
    """
    if method == "z-score":
        return (data - np.mean(data)) / np.std(data)
    elif method == "min-max":
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        raise ValueError("Invalid normalization method. Choose 'z-score' or 'min-max'.")

def extract_brain(data: np.ndarray, 
                  verbose: bool = True) -> np.ndarray:
    """
    Extract brain from a given image using deep learning brain extraction
    
    Args:
    data: np.ndarray: image data
    verbose: bool: whether to print the output
    
    Returns:
    np.ndarray: brain extracted image data
    """
    image = ants.from_numpy(data)
    mask = antspynet.brain_extraction(image, modality='t1', verbose=verbose)
    brain = image * mask
    return brain.numpy()

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

def crop_numpy(data, target_shape):
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
    """
    #data = get_data(image)
    resampled = resample_image(image, voxel_size=(1, 1, 1), output_format='numpy')
    normalized = normalize_data(resampled)
    extracted = extract_brain(normalized)
    cropped = crop_numpy(extracted, (192, 192, 192)) # TODO: fix cropping
    smoothed = smooth_gaussian(cropped) # TODO: fix smoothing
    return smoothed



