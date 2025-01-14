import matplotlib.pyplot as plt
import numpy as np

def plot_slices(data: np.ndarray, 
                title: str = "") -> None:
    """
    Plot 4 evenly spaced slices along the z-axis (axis 2) of the MRI volume.

    Parameters:
    - data (ndarray): 3D MRI volume.
    - title (str): Title of the plot.
    """
    z_dim = data.shape[2]  # Size along the z-axis
    slice_indices = np.linspace(0, z_dim - 1, 8, dtype=int)  # Select 5 slices evenly spaced
    slice_indices = slice_indices[2:-2]  # Remove first and last slices to avoid edge cases
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, slice_idx in enumerate(slice_indices):
        axes[i].imshow(data[:, :, slice_idx], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Slice {slice_idx}")
    
    fig.suptitle(title)
    plt.show()

def calculate_snr(data: np.ndarray) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) of the MRI volume.
    
    Parameters:
    - data (ndarray): 3D MRI volume.
    
    Returns:
    float: SNR value
    """
    signal = np.mean(data)
    noise = np.std(data)
    return signal / noise

def plot_histogram(data: np.ndarray, 
                   title="") -> None:
    """
    Plot the histogram of the MRI volume intensities.
    
    Parameters:
    - data (ndarray): 3D MRI volume.
    - title (str): Title of the plot
    """
    plt.hist(data.flatten(), bins=50, color="blue", alpha=0.7)
    plt.title(title)
    plt.show()

def validate_data_shape(data: np.ndarray, 
                        expected_shape: tuple) -> None:
    """
    Validate the shape of the input data.
    
    Parameters:
    - data (ndarray): Input data to validate.
    - expected_shape (tuple): Expected shape of the input data.
    """
    assert data.shape == expected_shape, f"Shape mismatch: {data.shape} != {expected_shape}"

