import os
import sys
from pathlib import Path

# Add the root directory to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from utils.preprocess import load_nii
from utils.preprocess_validation import plot_slices

"""
Results:
Passes
"""

def main(file_path):
    nii = load_nii(file_path)
    print(f'Data shape: {nii.shape}')
    plot_slices(data=nii, title="Loaded original data")

if __name__ == '__main__':
    # Get the current working directory of the script 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rel_file_path = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    file_path = os.path.join(current_dir, '..', rel_file_path)

    main(file_path)    