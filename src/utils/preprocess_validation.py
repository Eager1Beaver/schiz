import numpy as np
import nibabel as nib
from typing import Union
import matplotlib.pyplot as plt
from src.utils.preprocess import get_data
from skimage.filters import threshold_otsu
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

def plot_slices(data: Union[np.ndarray, nib.Nifti1Image], # TODO: added Nifti1Image support, redo docs // 
                how_many: int = 4,
                title: str = "",
                axes: list = None,
                isSave_fig: bool = False) -> None:
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
            if isSave_fig:
                fig.savefig("slices.pdf")

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
    
def calculate_snr(data: np.ndarray, eps: float = 1e-6) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) of the MRI volume.

    Warning: this method does not explicitly define signal (mean intensity in the brain region) 
    and noise (standard deviation in the background or non-brain region).
    
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
    
    #if data.ndim!= 3:
    #    raise ValueError(f"Input data must be a 3D numpy array, got {data.ndim}")
    
    if data.size == 0: 
        raise ValueError("Input data should not be empty")
    
    signal = np.mean(data)
    noise = np.std(data)

    if noise == 0: 
        raise ValueError("Standard deviation (noise) of the data is zero, SNR cannot be computed")
    
    return signal / max(noise, eps)

def visualize_signal_mask(data: np.ndarray, mask: np.ndarray, slice_index: int):
    """
    Visualize the original data and the generated mask for a specific slice.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Data")
    plt.imshow(data[:, :, slice_index], cmap="gray")
    
    plt.subplot(1, 2, 2)
    plt.title("Generated Mask")
    plt.imshow(mask[:, :, slice_index], cmap="gray")
    plt.show()

def generate_signal_mask(data: np.ndarray, otsu_scaling: float = 1.0, min_intensity_factor: float = 0.1) -> np.ndarray:
    """
    Combine Otsu's thresholding with an intensity-based threshold for mask generation.
    """
    # Filter non-zero values for Otsu's thresholding
    non_zero_data = data[data > 0]
    if non_zero_data.size == 0:
        print("Warning: No non-zero data found for thresholding. Returning a zero mask.")
        return np.zeros_like(data, dtype=np.uint8)
    
    otsu_threshold = threshold_otsu(data[data > 0]) * otsu_scaling
    intensity_threshold = min_intensity_factor * np.max(data)
    combined_threshold = max(otsu_threshold, intensity_threshold)
    return (data >= combined_threshold).astype(np.uint8)

def calculate_snr_with_mask(data: np.ndarray, mask = None, eps: float = 1e-6) -> float:
    """
    Calculate the SNR using a mask to focus on the brain region.

    Warning: This method explicitly defines signal (mean intensity in the brain region) 
    and noise (standard deviation in the background or non-brain region).
    
    Args:
        data (np.ndarray): Normalized 3D MRI volume.
        eps (float): Small value to prevent division by zero.
        
    Returns:
        float: SNR value.
    """
    if mask is None:
        mask = generate_signal_mask(data)
    #visualize_mask(data, mask, 130)

    # Check if the mask has any valid signal region
    if np.sum(mask) == 0:
        print("Warning: The mask contains no signal region. Assigning SNR value of 0.")
        return 0.0
    
    signal = np.mean(data[mask > 0])
    noise = np.std(data[mask == 0])
    
    if noise == 0:
        raise ValueError("Standard deviation (noise) is zero, SNR cannot be computed.")
    
    return signal / max(noise, eps)

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
    
    #if data1.ndim!= 3 or data2.ndim!= 3:
    #    raise ValueError(f"Input data must be 3D numpy arrays, got {data1.ndim} and {data2.ndim}")
    
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
    
    #if data1.ndim!= 3 or data2.ndim!= 3:
    #    raise ValueError(f"Input data must be 3D numpy arrays, got {data1.ndim} and {data2.ndim}")
    
    if data1.size == 0 or data2.size == 0: 
        raise ValueError("Input data should not be empty")
    
    if mse is None:  # If mse is not provided, calculate it
        mse = calculate_mse(data1, data2)

    if mse == 0:
        raise ValueError("Mean Squared Error (MSE) of the data is zero, PSNR cannot be computed")
    
    max_intensity = np.max(data1)
    if max_intensity == 0:
        raise ValueError("Maximum intensity value in the data is zero, PSNR cannot be computed")
    
    psnr = 20 * np.log10(max_intensity / np.sqrt(mse))
    return psnr

def calculate_relative_psnr_with_mask(data: np.ndarray, 
                                      mask: np.ndarray = None, 
                                      max_intensity: float = None, 
                                      eps: float = 1e-6) -> float:
    """
    Calculate a relative PSNR based on the difference between signal and noise, 
    focusing only on the regions defined by the mask.

    Warning: This method explicitly defines signal (mean intensity in the brain region)
    and noise (standard deviation in the background or non-brain region).
    
    Parameters:
    - data (np.ndarray): 3D MRI volume (already normalized).
    - mask (np.ndarray): Binary mask to define the signal region (e.g., brain).
    - max_intensity (float, optional): Maximum intensity value to use as a reference. Defaults to max value in the masked data.
    - eps (float, optional): Small value to avoid division by zero errors.

    Returns:
    - float: Relative PSNR value.
    """
    if mask is None:
        mask = generate_signal_mask(data)

    if not isinstance(data, np.ndarray) or not isinstance(mask, np.ndarray):
        raise TypeError(f"Inputs must be numpy arrays, got {type(data)} and {type(mask)}")
    
    if data.size == 0 or mask.size == 0:
        raise ValueError("Input data or mask cannot be empty")

    if data.shape != mask.shape:
        raise ValueError("Data and mask must have the same shape")

    # Define the signal and noise regions using the mask
    signal_region = data[mask > 0]
    noise_region = data[mask == 0]

    # Use the max intensity in the signal region if not provided
    if max_intensity is None:
        max_intensity = signal_region.max()

    # Calculate signal (mean of the signal region)
    signal = np.mean(signal_region)

    # Calculate noise (standard deviation of the noise region)
    noise = np.std(noise_region)

    if noise == 0:
        raise ValueError("Noise (standard deviation) is zero; PSNR cannot be computed.")

    # Compute Mean Squared Error (MSE) based on the signal and noise
    mse = np.mean((signal_region - signal)**2)

    # PSNR calculation
    psnr = 20 * np.log10(max_intensity / (np.sqrt(mse) + eps))  # Add eps to prevent log of zero

    return psnr

def calculate_relative_psnr(data: np.ndarray, max_intensity: float = None) -> float:
    """
    Calculate a relative PSNR using the maximum intensity value as the ideal reference.

    Warning: this method does not explicitly define signal (mean intensity in the brain region)
    and noise (standard deviation in the background or non-brain region)

    Parameters:
    - data (np.ndarray): 3D MRI volume.
    - max_intensity (float, optional): Maximum intensity value to use as a reference. 
                                       Defaults to the max value in the data.

    Returns:
    - float: Relative PSNR value.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Input must be a numpy array, got {type(data)}")
    
    if data.size == 0:
        raise ValueError("Input image cannot be empty")
    
    if max_intensity is None:
        max_intensity = data.max()
    
    mse = np.mean((data - data.mean())**2) # max_intensity
    if mse == 0:
        raise ValueError("Mean Squared Error (MSE) is zero; PSNR cannot be computed")
    
    psnr = 20 * np.log10(max_intensity / np.sqrt(mse))
    return psnr

def calculate_relative_psnr2(data: np.ndarray, max_intensity: float = None, eps: float = 1e-6) -> float:
    """
    Calculate a relative PSNR based on the difference between signal and noise, 
    with a focus on meaningful signal (brain regions in the case of MRI).
    
    Parameters:
    - data (np.ndarray): 3D MRI volume (already normalized).
    - max_intensity (float, optional): Maximum intensity value to use as a reference. Defaults to max value in the data.
    - eps (float, optional): Small value to avoid division by zero errors.

    Returns:
    - float: Relative PSNR value.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Input must be a numpy array, got {type(data)}")
    
    if data.size == 0:
        raise ValueError("Input image cannot be empty")
    
    if max_intensity is None:
        max_intensity = data.max()

    # Signal is calculated as the mean value of the data
    signal = np.mean(data)

    # Noise is the standard deviation of the data
    noise = np.std(data)

    if noise == 0:
        raise ValueError("Noise (standard deviation) is zero; PSNR cannot be computed.")

    # Compute Mean Squared Error (MSE) based on the signal and noise
    mse = np.mean((data - signal)**2)

    # PSNR calculation
    psnr = 20 * np.log10(max_intensity / (np.sqrt(mse) + eps))  # Add eps to prevent log of zero

    # Return PSNR value
    return psnr

def calculate_cnr(data: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Calculate the Contrast-to-Noise Ratio (CNR) of the MRI volume.
    
    Args:
        data (np.ndarray): Normalized 3D MRI volume.
        mask (np.ndarray): Binary mask indicating the brain region (1 for brain, 0 for background).
        
    Returns:
        float: CNR value.
    """
    
    if mask is None:
        mask = generate_signal_mask(data)

    # Signal: Mean intensity of the brain region (inside the mask)
    signal = np.mean(data[mask > 0])
    #print('cnr signal ', signal)
    
    # Background: Mean intensity of the non-brain region (outside the mask)
    background = np.mean(data[mask == 0])
    #print('cnr background ', background)
    
    # Noise: Standard deviation of the non-brain region (background)
    noise = np.std(data[mask == 0])
    #print('cnr noise ', noise)

    if noise == 0:
        #return 0
        raise ValueError("Noise (standard deviation of the background) is zero, CNR cannot be computed.")
    
    cnr = np.abs(signal - background) / noise
    return cnr

def calculate_rmse(data: np.ndarray, reference: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between the data and the reference.
    
    Args:
        data (np.ndarray): 3D MRI volume (e.g., the actual image).
        reference (np.ndarray): Ideal reference volume (e.g., maximum intensity).
        
    Returns:
        float: RMSE value.
    """
    if data.shape != reference.shape:
        raise ValueError("Data and reference must have the same shape.")
    
    rmse = np.sqrt(np.mean((data - reference) ** 2))
    return rmse

def calculate_relative_rmse(data: np.ndarray, max_intensity: float = None) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between the data and the reference (filled with max intensity).
    
    Args:
        data (np.ndarray): 3D MRI volume (e.g., the actual image).
        max_intensity (float): Maximum intensity for the reference.
        
    Returns:
        float: RMSE value.
    """
    if max_intensity is None:
        max_intensity = np.max(data)  # Use max intensity from data if not provided

    # Create reference image filled with max_intensity
    reference = np.full_like(data, max_intensity)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((data - reference) ** 2))
    return rmse

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
    
    #if data1.ndim!= 3 or data2.ndim!= 3:
    #    raise ValueError(f"Input data must be 3D numpy arrays, got {data1.ndim} and {data2.ndim}")
    
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
    
    #if data.ndim!= 3:
    #    raise ValueError(f"Input data must be a 3D numpy array, got {data.ndim}")
    
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
