import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def upsample_mcicshare(input_dir, output_dir):
    """Upsamples all NIfTI files in input_dir and saves them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(input_dir, filename)
            
            # Load the NIfTI image
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # Get original shape
            original_shape = data.shape
            print(f'Original shape: {original_shape}')
            
            # Compute zoom factors (only for the Z-dimension)
            z_factor = 192 / original_shape[1]  # Assuming slices are along the 2nd dimension (Y)
            zoom_factors = (1, z_factor, 1)  # Keep X and Z unchanged
            
            # Upsample using interpolation
            upsampled_data = zoom(data, zoom_factors, order=0)  # Nearest-neighbor interpolation
            
            # Create new NIfTI image
            upsampled_img = nib.Nifti1Image(upsampled_data, img.affine, img.header)
            print(f'Upsampled shape: {upsampled_img.shape}')
            
            # Save the upsampled image
            save_path = os.path.join(output_dir, filename)
            nib.save(upsampled_img, save_path)
            print(f'Upsampled and saved: {save_path}')
    
# Process train and test sets
upsample_mcicshare('data/MCICShare/train', 'data/MCICShare_upsampled/train')
upsample_mcicshare('data/MCICShare/test', 'data/MCICShare_upsampled/test')

print("Upsampling complete!")
