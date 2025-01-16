import numpy as np
import matplotlib.pyplot as plt

def plot_slices(data: np.ndarray,
                how_many: int = 4,
                title: str = "") -> None:
    """
    Plot 4 evenly spaced slices along the z-axis (axis 2) of the MRI volume (excluding the edge cases).

    Parameters:
    - data (ndarray): 3D MRI volume.
    - how_many (int): Number of slices to plot. Default is 4.
    - title (str): Title of the plot.

    Raises: ValueError: If the data is not 3D or `how_many` is invalid. 
    TypeError: If the input data is not a numpy array.
    """
    if not isinstance(data, np.ndarray): # Ensure data is a numpy array
        raise TypeError(f"Input data must be a numpy array, got {type(data)}")
    
    if data.ndim!= 3: # Ensure data is 3 dimensional
        raise ValueError(f"Input data must be a 3D numpy array, got {data.ndim}")
    
    if how_many < 1 or how_many * 2 > data.shape[2]:
        raise ValueError(f"Number of slices to plot must be between 1 and the total number of slices, got {how_many}")
    
    try:
        z_dim = data.shape[2]  # Size along the z-axis
        slice_indices = np.linspace(0, z_dim - 1, how_many * 2, dtype=int)  # Select evenly spaced slices
        #slice_indices = slice_indices[int(how_many/2):-int(how_many/2)]  # Remove first and last slices to avoid edge cases

        # Handle both odd and even values of how_many 
        start_index = (len(slice_indices) - how_many) // 2 
        end_index = start_index + how_many 
        slice_indices = slice_indices[start_index:end_index]

        if len(slice_indices) != how_many:
            raise ValueError("Calculated slice indices do not match `how_many`")
        
        fig, axes = plt.subplots(1, how_many, figsize=(20, 5))
        for i, slice_idx in enumerate(slice_indices):
            axes[i].imshow(data[:, :, slice_idx], cmap="gray")
            axes[i].axis("off")
            axes[i].set_title(f"Slice {slice_idx}")
        
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

def plot_histogram(data: np.ndarray, 
                   title="") -> None:
    """
    Plot the histogram of the MRI volume intensities.
    
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.hist(data.flatten(), bins=50, color="blue", alpha=0.7)
        ax.set_title(title)
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
