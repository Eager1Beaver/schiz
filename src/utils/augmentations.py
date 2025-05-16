import numpy as np
from scipy.ndimage import rotate, shift, affine_transform

def apply_translation(image: np.ndarray, max_shift: int = 5) -> np.ndarray:
    """
    Applies translation to an MRI image.

    Parameters:
    image (np.ndarray): The MRI image in array format.
    nax_shift (int): The translation offsets in the x and y directions.

    Returns:
    np.ndarray: The translated MRI image.
    """
    shift_val = np.random.choice([-max_shift, max_shift])
    #print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got {image.ndim}D")
    if any(dim == 0 for dim in image.shape):
        raise ValueError(f"Image has zero dimension: {image.shape}")
    #return shift(image, shift=(0, shift_val, shift_val), mode='nearest')
    return shift(image, shift=(shift_val, shift_val, 0), mode='nearest')

def apply_rotation(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Rotates an MRI image by a given angle.

    Parameters:
    image (np.ndarray): The MRI image in array format.
    max_angle (float): The angle of rotation in degrees.

    Returns:
    np.ndarray: The rotated MRI image.
    """
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate(image, angle, axes=(0, 1), reshape=False, mode='nearest') # 1,2

def apply_gaussian_noise(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Adds Gaussian noise to an MRI image.

    Parameters:
    image (np.ndarray): The MRI image in array format.
    mean (float): The mean of the Gaussian noise.
    std (float): The standard deviation factor for the Gaussian noise.

    Returns:
    np.ndarray: The MRI image with added Gaussian noise.
    """
    std_g = max(image.std() * std, 1e-6)
    noise = np.random.normal(mean, std_g, image.shape).astype(image.dtype)
    return image + noise

def apply_shearing(image: np.ndarray, max_shear: float = 0.1) -> np.ndarray:
    """
    Applies shearing to an MRI image.

    Parameters:
    image (np.ndarray): The MRI image in array format.
    shear_level (float): The shearing level (small values like 0.05–0.2 are recommended).

    Returns:
    np.ndarray: The sheared MRI image.
    """
    shear_level = np.random.uniform(-max_shear, max_shear)
    shear_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, shear_level, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    affine = shear_matrix[:3, :3]
    offset = shear_matrix[:3, 3]
    return affine_transform(image, matrix=affine, offset=offset, order=1, mode='nearest')

def apply_contrast_adjustment(image: np.ndarray, method: str = 'gamma', min_factor=0.5, max_factor=1.5) -> np.ndarray:
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
    norm_img = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)
    factor = np.random.uniform(min_factor, max_factor)

    if method == 'gamma':
        adjusted = np.power(norm_img, factor)
    elif method == 'linear':
        adjusted = np.clip(0.5 + factor * (norm_img - 0.5), 0, 1)
    else:
        raise ValueError("Invalid method. Use 'gamma' or 'linear'.")

    adjusted = adjusted * (np.max(image) - np.min(image) + 1e-6) + np.min(image)
    return adjusted.astype(image.dtype)
