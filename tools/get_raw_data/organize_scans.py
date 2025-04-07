import os
import shutil
def extract_subject_id(filename):
    """Extracts subject ID from a filename like sub-A00000909_ses-20110101_acq-mprage_run-01_echo-01_T1w.pt"""
    return filename.split('_')[0]  # Extracts 'sub-A00000909'

def find_original_scan(subject_id, base_dir):
    """Searches for the original scan in the given base directory."""
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith(subject_id) and file.endswith(".nii.gz"):
                return os.path.join(root, file)
    return None

def organize_scans(modified_dir, original_base_dir, output_dir):
    """Finds and copies the original scans to the structured output directories."""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(modified_dir):
        if filename.endswith(".pt"):
            subject_id = extract_subject_id(filename)
            original_scan_path = find_original_scan(subject_id, original_base_dir)
            
            if original_scan_path:
                shutil.copy(original_scan_path, os.path.join(output_dir, os.path.basename(original_scan_path)))
                print(f"Copied {original_scan_path} -> {output_dir}")
            else:
                print(f"Warning: Original scan for {subject_id} not found!")

# Define paths
modified_dirs = {
    "COBRE": {"train": "data/dataset_2/train", "test": "data/dataset_2/test"},
    "MCICShare": {"train": "data/dataset_2/train", "test": "data/dataset_2/test"}
}
original_dirs = {
    "COBRE": "data/schizconnect_COBRE_images_22613/COBRE",
    "MCICShare": "data/schizconnect_MCICShare_images_22613/MCICShare"
}
output_dirs = {
    "COBRE": {"train": "data/COBRE/train", "test": "data/COBRE/test"},
    "MCICShare": {"train": "data/MCICShare/train", "test": "data/MCICShare/test"}
}

# Process each dataset
datasets = ["COBRE", "MCICShare"]
for dataset in datasets:
    for split in ["train", "test"]:
        organize_scans(modified_dirs[dataset][split], original_dirs[dataset], output_dirs[dataset][split])
