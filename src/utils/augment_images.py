import os
import torch
import random
from logger import logger
from utils.augmentations import apply_translation, apply_rotation, apply_gaussian_noise

def augment_images(
        augmentations: list[tuple],
        num_augmentations: int = 3,
        raw_nii_dir: str = "data/raw_nii",
        train_set_dir: str = "train_set",
        preprocessed_train_set_dir: str = "preprocessed",
        output_dir="augmented") -> None:
    """
    Applies user-specified augmentations to images in input_dir and saves them to output_dir.

    Parameters:
    - augmentations (list of tuples): List of augmentations to apply. Each tuple should be (augmentation_type, params).
      Example: [("translation", {"shift": (-5, 5)}), ("rotation", {"angle": (-10, 10)})]
    - num_augmentations (int): Number of times to apply the augmentation sequence per image.
    - input_dir (str): Path to the input directory containing .pt images (images converted to Pytorch tensors).
    - output_dir (str): Path to save augmented images.

    Returns:
        None
    """
    input_dir = os.path.join(raw_nii_dir, train_set_dir, preprocessed_train_set_dir)
    output_dir = os.path.join(input_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get images from folder and load them
    for file in os.listdir(input_dir):
        if file.endswith(".pt"):
            file_path = os.path.join(input_dir, file)
            image_data = torch.load(file_path).numpy()

            # Make a copy of the data for each augmentation pass
            for i in range(num_augmentations):
                augmented_data = image_data.copy()
                augmentation_description = []

                # Apply each augmentation in sequence
                for aug_type, params in augmentations:
                    if aug_type == "translation":
                        shift_x = random.uniform(*params["shift"])
                        shift_y = random.uniform(*params["shift"])
                        augmented_data = apply_translation(
                            augmented_data, (shift_x, shift_y))
                        augmentation_description.append(
                            f"trans_{shift_x:.1f}_{shift_y:.1f}")

                    elif aug_type == "rotation":
                        angle = random.uniform(*params["angle"])
                        augmented_data = apply_rotation(augmented_data, angle)
                        augmentation_description.append(f"rot_{angle:.1f}")

                    elif aug_type == "gaussian_noise":
                        mean = params.get("mean", 0)
                        std = params["std"]
                        augmented_data = apply_gaussian_noise(
                            augmented_data, mean, std)
                        augmentation_description.append(f"noise_{std:.2f}")

                # Save augmented image with detailed filename
                save_name = f"{os.path.splitext(file)[0]}_{'_'.join(augmentation_description)}_{i}.pt"
                save_path = os.path.join(output_dir, save_name)
                torch.save(torch.tensor(augmented_data, dtype=torch.float32),
                           save_path)
                
                logger.info(f"Saved: {save_path}")
