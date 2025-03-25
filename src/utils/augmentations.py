import numpy as np
from scipy.ndimage import rotate, shift


def apply_translation(image: np.ndarray,
                      translation: tuple[float, float]) -> np.ndarray:
    """
    Applies translation to an MRI image.

    Parameters:
    image (np.ndarray): The MRI image in array format.
    translation (tuple[float, float]): The translation offsets in the x and y directions.

    Returns:
    np.ndarray: The translated MRI image.
    """
    return shift(image,
                 shift=(translation[0], translation[1], 0),
                 mode='nearest')


def apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an MRI image by a given angle.

    Parameters:
    image (np.ndarray): The MRI image in array format.
    angle (float): The angle of rotation in degrees.

    Returns:
    np.ndarray: The rotated MRI image.
    """
    return rotate(image, angle, axes=(0, 1), reshape=False, mode='nearest')


def apply_gaussian_noise(image: np.ndarray, mean: float,
                         std: float) -> np.ndarray:
    """
    Adds Gaussian noise to an MRI image.

    Parameters:
    image (np.ndarray): The MRI image in array format.
    mean (float): The mean of the Gaussian noise.
    std (float): The standard deviation factor for the Gaussian noise.

    Returns:
    np.ndarray: The MRI image with added Gaussian noise.
    """
    std_g = max(image.std() * std, 1e-6)  # Prevents std from being too small
    noise = np.random.normal(mean, std_g, image.shape).astype(image.dtype)
    return image + noise
