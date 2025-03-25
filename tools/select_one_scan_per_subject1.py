import pandas as pd

# Load the CSV file
path_to_quality_check = "/home/user/projects/schiz/tools/initial_quality_check_results/quality_check_all_results_COBRE.csv" # quality_check_COBRE.csv
#path_to_quality_check = "quality_check_COBRE.csv"
df = pd.read_csv(path_to_quality_check)

# Extract subject ID from file path
df["subject_id"] = df["file_path"].str.extract(r"(sub-A\d+)")

# Keep only T1 Scans
df = df[df["file_path"].str.contains("T1")]

# Define the desired SNR range
min_snr, max_snr, target_snr = 8.8, 9.2, 9.0

selected_scans = []

for subject, group in df.groupby("subject_id"):
    # Filter scans within the desired SNR range
    filtered = group[(group["snr"] >= min_snr) & (group["snr"] <= max_snr)]
    
    if not filtered.empty:
        # Select the scan closest to the target SNR of 9.0
        best_scan = filtered.iloc[(filtered["snr"] - target_snr).abs().argmin()]
    else:
        # If no scan fits in the range, pick the one closest to 9.0
        best_scan = group.iloc[(group["snr"] - target_snr).abs().argmin()]
    
    selected_scans.append(best_scan)

# Create a new DataFrame for the selected scans
selected_df = pd.DataFrame(selected_scans)

# Drop the temporary subject_id column
selected_df = selected_df.drop(columns=["subject_id"])

# Save to a new CSV file
selected_df.to_csv("selected_scans_COBRE.csv", index=False)
