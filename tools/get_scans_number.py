import os
import glob
import pandas as pd

def get_paths_labels(data_path):
    """
    Get paths to all.nii.gz files and their corresponding labels.

    Args:
    data_path: str: path to the top-level "data" folder

    Returns:
    file_paths: list: list of paths to.nii.gz files
    labels: list: list of labels (0 or 1) corresponding to each file (each subject)

    """
    # Initialize empty lists to store file paths and labels
    file_paths = []
    labels = []

    # Iterate over COBRE and MCICShare folders
    for dataset_name in ["COBRE", "MCICShare"]:
        dataset_full_name = "schizconnect_" + dataset_name + "_images" + "_22613"
        dataset_path = os.path.join(data_path, dataset_full_name, dataset_name)
        csv_path = os.path.join(dataset_path, "participants.csv")

        # Load participant mapping from CSV
        participants_df = pd.read_csv(csv_path)
        participants_df.set_index(participants_df["participant_id"].str.replace("sub-", "", regex=False), inplace=True)

        # Find all .nii.gz files in the anat subfolder
        anat_files = glob.glob(os.path.join(dataset_path, "sub-*", "ses-*", "anat", "*.nii.gz"))

        for file_path in anat_files:
            # Extract subject ID (e.g., "A00000300" from "sub-A00000300")
            subject_id = os.path.basename(file_path).split("_")[0].replace("sub-", "")

            # Retrieve the label (0 or 1) from the CSV
            if subject_id in participants_df.index:
                label = participants_df.loc[subject_id, "dx_encoded"]
                file_paths.append(file_path)
                labels.append(label)
            else: # TODO: collect subjects with wrong labels or mismatches?
                print(f"Warning: No label found for {subject_id} in {csv_path}")

    return file_paths, labels            

def main(labels):
    number_of_schiz_scans = labels.count(1)
    number_of_healthy_scans = labels.count(0)

    return number_of_schiz_scans, number_of_healthy_scans

if __name__ == '__main__':
    data_path = "/home/user/projects/schiz/data" #'data' #/home/user/projects/schiz/data
    file_paths, labels = get_paths_labels(data_path)
    schiz_scans_num, healthy_scans_num = main(labels)

    print(f"Number of Schizophrenia scans: {schiz_scans_num}")
    print(f"Number of Healthy scans: {healthy_scans_num}")  