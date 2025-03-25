import pandas as pd
import numpy as np

# Load the CSV file
path_to_quality_check = "/home/user/projects/schiz/tools/initial_quality_check_results/quality_check_all_results_COBRE.csv" # quality_check_COBRE.csv
df = pd.read_csv(path_to_quality_check)

# Extract subject ID from file path
df["subject_id"] = df["file_path"].str.extract(r"(sub-A\d+)")

# Define a range of potential target SNR values to test
snr_values = np.linspace(df["snr"].quantile(0.25), df["snr"].quantile(0.75), num=50)

best_target_snr = None
best_selected_scans = None
best_spread = float("inf")

# Try different target SNR values and pick the one that minimizes spread
for target_snr in snr_values:
    selected_scans = []
    for subject, group in df.groupby("subject_id"):
        best_scan = group.iloc[(group["snr"] - target_snr).abs().argmin()]
        selected_scans.append(best_scan)
    
    selected_df = pd.DataFrame(selected_scans)
    spread = selected_df["snr"].std()
    
    if spread < best_spread:
        best_spread = spread
        best_target_snr = target_snr
        best_selected_scans = selected_df

# Drop the temporary subject_id column
best_selected_scans = best_selected_scans.drop(columns=["subject_id"])

# Save to a new CSV file
best_selected_scans.to_csv("selected_scans_COBRE.csv", index=False)
