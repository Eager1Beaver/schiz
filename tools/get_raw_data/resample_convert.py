import os
import torch
import nibabel as nib
from utils.preprocess import resample_image

# Define input and output directories
input_dirs = {
    "train": ["data/COBRE/train", "data/MCICShare_upsampled/train"],
    "test": ["data/COBRE/test", "data/MCICShare_upsampled/test"]
}

output_dirs = {
    "train": "data/dataset_1/train",
    "test": "data/dataset_1/test"
}

# Ensure output directories exist
for out_dir in output_dirs.values():
    os.makedirs(out_dir, exist_ok=True)

# Process each dataset
for split, paths in input_dirs.items():
    output_dir = output_dirs[split]

    for input_dir in paths:
        for filename in os.listdir(input_dir):
            if filename.endswith(".nii.gz"):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename.replace(".nii.gz", ".pt"))

                # Load the NIfTI image
                img = nib.load(input_path)

                # Resample to (2,2,2) voxel size
                resampled_data = resample_image(img, voxel_size=(2, 2, 2), output_format="numpy")

                # Convert to PyTorch tensor and save
                torch.save(torch.tensor(resampled_data, dtype=torch.float32), output_path)

                print(f"Processed and saved: {output_path}")

print("Resampling and conversion complete!")
