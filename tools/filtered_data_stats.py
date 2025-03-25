import os
import pandas as pd

# Load subject details
subject_details = pd.read_csv("subject_scan_details.csv")

# Create a dictionary mapping Subject ID to their details
subject_dict = subject_details.set_index("Subject ID").to_dict(orient="index")

# Define the base data folders and dataset names
datasets = {"data/COBRE": "COBRE", "data/MCICShare": "MCICShare"}

# Collect scan data
scan_data = []

for folder, dataset_name in datasets.items():
    for file in os.listdir(folder):
        if file.endswith(".pt"):
            # Extract subject ID from the filename
            subject_id = file.split("_")[0].replace("sub-", "")  # Extract subject ID
            
            # Check if the subject exists in the CSV
            if subject_id in subject_dict:
                details = subject_dict[subject_id]
                scan_data.append([
                    os.path.join(folder, file),  # Full file path
                    dataset_name,  # Dataset name (COBRE or MCICShare)
                    subject_id,
                    details["Schizophrenic"],
                    details["Age"],
                    details["Sex"]
                ])

# Create DataFrame
columns = ["file_path", "dataset", "subject_id", "schizophrenic", "age", "sex"]
scan_df = pd.DataFrame(scan_data, columns=columns)

# Save to CSV
scan_df.to_csv("subject_scan_info.csv", index=False)

print("CSV file created: subject_scan_info.csv")
