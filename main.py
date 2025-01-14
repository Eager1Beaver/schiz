from src.data_loader import MRIDataset

from src.preprocess_validation import plot_slices

def main():
    # Load the data
    path_sample_1 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_acq-mprage_run-01_T1w.nii.gz"
    path_sample_2 = "data/schizconnect_COBRE_images_22613/COBRE/sub-A00000300/ses-20110101/anat/sub-A00000300_ses-20110101_T2w.nii.gz"

    # TODO: 
    # only a path to the data folder is supposed to be provided
    # the rest is supposed to be handled by the data_loader.py
    file_paths = [path_sample_1, path_sample_2]

    # Create the dataset, labels are dummy values, not used
    # TODO: labels are supposed to be loaded from a csv file
    dataset = MRIDataset(file_paths, labels=[0, 1])

    # Get the first sample
    sample_1 = dataset[0]
    print('sample_1 loaded successfully')

    # Plot the slices of the first sample
    plot_slices(sample_1[0], title="Sample 1")

    # Get the second sample
    sample_2 = dataset[1]
    print('sample_2 loaded successfully')

    # Plot the slices of the second sample
    plot_slices(sample_2[0], title="Sample 2")

if __name__ == '__main__':
    main()