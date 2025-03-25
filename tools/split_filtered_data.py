import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Load subject scan info
df = pd.read_csv("subject_scan_info.csv")

# Ensure baseline_data/train and baseline_data/test directories exist
train_dir = "baseline_data/train"
test_dir = "baseline_data/test"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Separate scans into schizophrenic and healthy groups
schiz_scans = df[df["schizophrenic"] == True]
healthy_scans = df[df["schizophrenic"] == False]

# Perform stratified 80/20 split within each group
schiz_train, schiz_test = train_test_split(schiz_scans, test_size=0.2, random_state=42, shuffle=True)
healthy_train, healthy_test = train_test_split(healthy_scans, test_size=0.2, random_state=42, shuffle=True)

# Combine train and test sets
train_set = pd.concat([schiz_train, healthy_train])
test_set = pd.concat([schiz_test, healthy_test])

# Function to copy files to respective folders
def copy_files(file_list, destination_dir):
    for _, row in file_list.iterrows():
        src_path = row["file_path"]
        dest_path = os.path.join(destination_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)  # Copy with metadata

# Copy files into respective directories
copy_files(train_set, train_dir)
copy_files(test_set, test_dir)

print("Dataset successfully split and copied!")