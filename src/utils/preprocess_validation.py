import numpy as np
import nibabel as nib
from typing import Union
import matplotlib.pyplot as plt
from src.utils.preprocess import get_data
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

def plot_slices(data: Union[np.ndarray, nib.Nifti1Image], # TODO: added Nifti1Image support, redo docs 
                how_many: int = 4,
                title: str = "",
                axes: list = None) -> None:
    """
    Plot 4 evenly spaced slices along the z-axis (axis 2) of the MRI volume (excluding the edge cases).
    Plot slices on provided axes or create a new figure if axes are not provided.

    Parameters:
    - data (ndarray): 3D MRI volume.
    - how_many (int): Number of slices to plot. Default is 4.
    - title (str): Title of the plot.

    Raises: ValueError: If the data is not 3D or `how_many` is invalid. 
    TypeError: If the input data is not a numpy array.
    """
    if not isinstance(data, np.ndarray) or not isinstance(data, nib.Nifti1Image): # Ensure data is a numpy array # 
        raise TypeError(f"Input data must be a numpy array or a NIfTI image, got {type(data)}")
    
    if data.ndim!= 3: # Ensure data is 3 dimensional
        raise ValueError(f"Input data must be a 3D numpy array, got {data.ndim}")
    
    if how_many < 1 or how_many * 2 > data.shape[2]:
        raise ValueError(f"Number of slices to plot must be between 1 and the total number of slices, got {how_many}")
    
    try:
        if not isinstance(data, np.ndarray):
            data = get_data(data)
        z_dim = data.shape[2]  # Size along the z-axis
        slice_indices = np.linspace(0, z_dim - 1, how_many * 2, dtype=int)  # Select evenly spaced slices
        
        # Handle both odd and even values of how_many 
        start_index = (len(slice_indices) - how_many) // 2 
        end_index = start_index + how_many 
        slice_indices = slice_indices[start_index:end_index]

        if len(slice_indices) != how_many:
            raise ValueError("Calculated slice indices do not match `how_many`")
        
        if axes is None:
            fig, axes = plt.subplots(1, how_many, figsize=(20, 5))
            own_axes = True
        else:
            own_axes = False    

        for i, slice_idx in enumerate(slice_indices):
            axes[i].imshow(data[:, :, slice_idx], cmap="gray")
            axes[i].axis("off")
            axes[i].set_title(f"Slice {slice_idx}")
        
        if own_axes:
            fig.suptitle(title)
            plt.show()

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
    
def calculate_snr(data: np.ndarray) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) of the MRI volume.
    
    Parameters:
    - data (ndarray): 3D MRI volume.
    
    Returns:
    float: SNR value

    Raises:
    ValueError: If the data is not 3D or empty.
    TypeError: If the input data is not a numpy array.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Input data must be a numpy array, got {type(data)}")
    
    if data.ndim!= 3:
        raise ValueError(f"Input data must be a 3D numpy array, got {data.ndim}")
    
    if data.size == 0: 
        raise ValueError("Input data should not be empty")
    
    signal = np.mean(data)
    noise = np.std(data)

    if noise == 0: 
        raise ValueError("Standard deviation (noise) of the data is zero, SNR cannot be computed")
    
    return signal / noise

def calculate_mse(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between two MRI volumes.
    
    Parameters:
    - data1 (ndarray): 3D MRI volume.
    - data2 (ndarray): 3D MRI volume.
    
    Returns:
    float: MSE value

    Raises:
    ValueError: If the data is not 3D or empty.
    TypeError: If the input data is not a numpy array.
    """
    if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
        raise TypeError(f"Input data must be numpy arrays, got {type(data1)} and {type(data2)}")
    
    if data1.ndim!= 3 or data2.ndim!= 3:
        raise ValueError(f"Input data must be 3D numpy arrays, got {data1.ndim} and {data2.ndim}")
    
    if data1.size == 0 or data2.size == 0: 
        raise ValueError("Input data should not be empty")
    
    return mean_squared_error(data1.flatten(), data2.flatten())

def calculate_psnr(data1: np.ndarray, 
                   data2: np.ndarray,
                   mse: float = None) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two MRI volumes.
    
    Parameters:
    - data1 (ndarray): 3D MRI volume.
    - data2 (ndarray): 3D MRI volume.

    Returns:
    float: PSNR value

    Raises:
    ValueError: If the data is not 3D or empty.
    TypeError: If the input data is not a numpy array.
    """                                                                             
    if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
        raise TypeError(f"Input data must be numpy arrays, got {type(data1)} and {type(data2)}")
    
    if data1.ndim!= 3 or data2.ndim!= 3:
        raise ValueError(f"Input data must be 3D numpy arrays, got {data1.ndim} and {data2.ndim}")
    
    if data1.size == 0 or data2.size == 0: 
        raise ValueError("Input data should not be empty")
    
    if mse is None:  # If mse is not provided, calculate it
        mse = calculate_mse(data1, data2)

    if mse == 0:
        raise ValueError("Mean Squared Error (MSE) of the data is zero, PSNR cannot be computed")
    
    max_val = np.max(data1)
    if max_val == 0:
        raise ValueError("Maximum intensity value in the data is zero, PSNR cannot be computed")
    
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def calculate_ssim(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between two MRI volumes.
    
    Parameters:
    - data1 (ndarray): 3D MRI volume.
    - data2 (ndarray): 3D MRI volume.
    
    Returns:
    float: SSIM value

    Raises:
    ValueError: If the data is not 3D or empty.
    TypeError: If the input data is not a numpy array.
    """
    if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
        raise TypeError(f"Input data must be numpy arrays, got {type(data1)} and {type(data2)}")
    
    if data1.ndim!= 3 or data2.ndim!= 3:
        raise ValueError(f"Input data must be 3D numpy arrays, got {data1.ndim} and {data2.ndim}")
    
    if data1.size == 0 or data2.size == 0: 
        raise ValueError("Input data should not be empty")
    
    return ssim(data1, data2, data_range=data1.max() - data1.min())

def plot_histogram(data: np.ndarray, 
                   title="",
                   ax=None) -> None:
    """
    Plot the histogram of the MRI volume intensities.
    Plot the histogram on the provided axis or create a new figure if none is provided.
    
    Parameters:
    - data (ndarray): 3D MRI volume.
    - title (str): Title of the plot

    Raises:
    ValueError: If the data is not 3D or empty.
    TypeError: If the input data is not a numpy array.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Input data must be a numpy array, got {type(data)}")
    
    if data.ndim!= 3:
        raise ValueError(f"Input data must be a 3D numpy array, got {data.ndim}")
    
    if data.size == 0: 
        raise ValueError("Input data should not be empty")
    
    try:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.hist(data.flatten(), bins=50, color="blue", alpha=0.7)
        ax.set_title(title)
        if not ax:
            plt.show()

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

def validate_data_shape(data: np.ndarray, 
                        expected_shape: tuple) -> None:
    """
    Validate the shape of the input data.
    
    Parameters:
    - data (ndarray): Input data to validate.
    - expected_shape (tuple): Expected shape of the input data.
    """
    assert data.shape == expected_shape, f"Shape mismatch: {data.shape} != {expected_shape}"
