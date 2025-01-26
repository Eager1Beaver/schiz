import os
import glob
import pandas as pd
from collections import defaultdict

def get_paths_labels(data_path):
    """
    Get paths to all .nii.gz files and their corresponding labels.

    Args:
    data_path: str: path to the top-level "data" folder

    Returns:
    file_paths: list: list of paths to .nii.gz files
    labels: list: list of labels (0 or 1) corresponding to each file (each subject)
    file_info: dict: contains additional information for analysis
    """
    # Initialize storage for results
    file_paths = []
    labels = []
    file_info = defaultdict(lambda: defaultdict(list))  # Nested dictionary

    # Iterate over COBRE and MCICShare folders
    for dataset_name in ["COBRE", "MCICShare"]:
        dataset_full_name = "schizconnect_" + dataset_name + "_images_22613"
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

                # Determine the scan type (T1 or T2)
                if "_T1w.nii.gz" in file_path:
                    scan_type = "T1"
                elif "_T2w.nii.gz" in file_path:
                    scan_type = "T2"
                else:
                    scan_type = "Other"

                # Update file_info dictionary
                dataset_key = dataset_name
                file_info[dataset_key][subject_id].append((scan_type, file_path, label))
            else:
                print(f"Warning: No label found for {subject_id} in {csv_path}")

    return file_paths, labels, file_info

def analyze_datasets(file_info):
    """
    Calculate the number of schizophrenic and healthy scans for each dataset.

    Args:
    file_info: dict: contains dataset, subject, and scan type information

    Returns:
    dataset_summary: list of dicts summarizing schizophrenic and healthy scans for each dataset
    """
    dataset_summary = []

    for dataset, subjects in file_info.items():
        schiz_scans = 0
        healthy_scans = 0

        for scans in subjects.values():
            for scan in scans:
                if scan[2] == 1:
                    schiz_scans += 1
                elif scan[2] == 0:
                    healthy_scans += 1

        dataset_summary.append({
            "Dataset": dataset,
            "Schizophrenic Scans": schiz_scans,
            "Healthy Scans": healthy_scans
        })

    return pd.DataFrame(dataset_summary)

def analyze_subjects(file_info):
    """
    Calculate the number of subjects and their labels for each dataset.

    Args:
    file_info: dict: contains dataset, subject, and scan type information

    Returns:
    subject_summary: list of dicts summarizing the number of subjects for each dataset
    """
    subject_summary = []

    for dataset, subjects in file_info.items():
        schiz_subjects = sum(1 for scans in subjects.values() if scans[0][2] == 1)
        healthy_subjects = sum(1 for scans in subjects.values() if scans[0][2] == 0)

        subject_summary.append({
            "Dataset": dataset,
            "Number of Subjects": len(subjects),
            "Schizophrenic Subjects": schiz_subjects,
            "Healthy Subjects": healthy_subjects
        })

    return pd.DataFrame(subject_summary)

def analyze_scans(file_info):
    """
    Calculate the number of scans for each patient, including T1 and T2 scans, and mark schizophrenia status.

    Args:
    file_info: dict: contains dataset, subject, and scan type information

    Returns:
    scan_summary: list of dicts with detailed scan information for each patient
    """
    scan_summary = []

    for dataset, subjects in file_info.items():
        for subject_id, scans in subjects.items():
            t1_scans = sum(1 for scan in scans if scan[0] == "T1")
            t2_scans = sum(1 for scan in scans if scan[0] == "T2")
            label = scans[0][2] if scans else None  # Assuming all scans for a subject have the same label

            scan_summary.append({
                "Dataset": dataset,
                "Subject ID": subject_id,
                "T1 Scans": t1_scans,
                "T2 Scans": t2_scans,
                "Schizophrenic": label == 1
            })

    return pd.DataFrame(scan_summary)

def main():
    data_path = "/home/user/projects/schiz/data"  # Adjust the path as needed
    file_paths, labels, file_info = get_paths_labels(data_path)

    # Analyze and save dataset summary
    dataset_summary_df = analyze_datasets(file_info)
    dataset_summary_csv_path = os.path.join(data_path, "dataset_scan_summary.csv")
    dataset_summary_df.to_csv(dataset_summary_csv_path, index=False)

    # Analyze and save subject summary
    subject_summary_df = analyze_subjects(file_info)
    subject_summary_csv_path = os.path.join(data_path, "dataset_subject_summary.csv")
    subject_summary_df.to_csv(subject_summary_csv_path, index=False)

    # Analyze and save scan summary
    scan_summary_df = analyze_scans(file_info)
    scan_summary_csv_path = os.path.join(data_path, "subject_scan_details.csv")
    scan_summary_df.to_csv(scan_summary_csv_path, index=False)

    print(f"Analysis complete. Results saved to the following files:\n"
          f"- {dataset_summary_csv_path}\n"
          f"- {subject_summary_csv_path}\n"
          f"- {scan_summary_csv_path}")

if __name__ == '__main__':
    main()
