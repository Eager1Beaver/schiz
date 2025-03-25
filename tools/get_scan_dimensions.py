import pandas as pd
import nibabel as nib

# Load the CSV file
path_to_quality_check = "/home/user/projects/schiz/tools/initial_quality_check_results/quality_check_all_results_COBRE.csv" # quality_check_COBRE.csv
df = pd.read_csv(path_to_quality_check)

# Prepare a list to store scan dimensions
dimensions = []

counter = 0
for file_path in df["file_path"]:
    try:
        # Load the MRI scan
        img = nib.load(file_path)
        x_dim, y_dim, z_dim = img.shape[:3]  # Extract spatial dimensions
        dimensions.append([file_path, x_dim, y_dim, z_dim])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        dimensions.append([file_path, None, None, None])  # Handle errors gracefully
    counter += 1    
    print(f'Processed {counter} files')    

# Create a DataFrame with the results
dim_df = pd.DataFrame(dimensions, columns=["file_path", "x_dim", "y_dim", "z_dim"])

# Save to a new CSV file
dim_df.to_csv("scan_dimensions_COBRE.csv", index=False)
