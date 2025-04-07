import numpy as np
from scipy.ndimage import rotate, shift, affine_transform


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

def apply_shearing(image: np.ndarray, shear_level: float) -> np.ndarray:
    """
    Applies shearing to an MRI image.

    Parameters:
    image (np.ndarray): The MRI image in array format.
    shear_level (float): The shearing level (small values like 0.05–0.2 are recommended).

    Returns:
    np.ndarray: The sheared MRI image.
    """
    shear_matrix = np.array([
        [1, shear_level, 0, 0],
        [0, 1,            0, 0],
        [0, 0,            1, 0],
        [0, 0,            0, 1]
    ])
    affine = shear_matrix[:3, :3]
    offset = shear_matrix[:3, 3]
    return affine_transform(image, matrix=affine, offset=offset, order=1, mode='nearest')


def apply_contrast_adjustment(image: np.ndarray, method: str = 'gamma', factor: float = 1.0) -> np.ndarray:
    """
    Adjusts contrast of an MRI image using gamma correction or linear contrast scaling.

    Parameters:
    image (np.ndarray): The MRI image in array format.
    method (str): 'gamma' for gamma correction, 'linear' for linear contrast scaling.
    factor (float): Contrast adjustment factor (e.g., gamma value or linear scaling factor).

    Returns:
    np.ndarray: The contrast-adjusted MRI image.
    """
    image = image.astype(np.float32)
    norm_img = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)  # normalize to [0,1]

    if method == 'gamma':
        adjusted = np.power(norm_img, factor)
    elif method == 'linear':
        adjusted = np.clip(0.5 + factor * (norm_img - 0.5), 0, 1)
    else:
        raise ValueError("Invalid method. Use 'gamma' or 'linear'.")

    # Restore original scale
    adjusted = adjusted * (np.max(image) - np.min(image) + 1e-6) + np.min(image)
    return adjusted.astype(image.dtype)
