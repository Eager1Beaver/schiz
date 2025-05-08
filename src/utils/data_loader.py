import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from skimage.transform import resize
import joblib


def extract_subject_number(file_name: str) -> str:
    """
    Extracts the subject identifier from a given file name.

    Parameter:
    file_name (str): The file name containing the subject identifier.

    Returns:
    str: The extracted subject identifier (e.g., 'A123'), or None if no match is found.
    """

    match = re.search(r'sub-(A\d+)', file_name)
    if match:
        return match.group(1)
    return None


PLOT_DIR = 'plots'
MODEL_DIR = 'models'
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

train_path = "train_path"
test_path = "test_path"
label_file = "participants.csv"


def load_middle_slices(
    npz_path: str, target_shape: tuple = (128, 128)) -> np.ndarray:
    """
    Loads a 3D MRI volume from a .npz file, extracts the middle 70% of slices, 
    normalizes them, resizes each slice to the specified target shape, and 
    returns them as a 3D numpy array.

    Parameters:
    npz_path (str): Path to the .npz file containing the 3D MRI volume.
    target_shape (tuple): The target height and width for resizing each 2D slice. 
                          Default is (128, 128).

    Returns:
    np.ndarray: A 3D numpy array of the processed 2D slices, with the shape 
                (num_slices, target_shape[0], target_shape[1]).
    """
    data = np.load(npz_path)
    array = data['arr_0']  # or whatever key the array is stored under

    # Ensure correct shape (slices, height, width)
    if array.shape[0] < array.shape[1] and array.shape[0] < array.shape[2]:
        data = np.transpose(array, (0, 1, 2))
    elif array.shape[1] < array.shape[0] and array.shape[1] < array.shape[2]:
        data = np.transpose(array, (1, 0, 2))
    elif array.shape[2] < array.shape[0] and array.shape[2] < array.shape[1]:
        data = np.transpose(array, (2, 0, 1))
    else:
        data = array  # no need to transpose if already (slices, height, width)

    num_slices = data.shape[0]
    start = int(num_slices * 0.15)
    end = int(num_slices * 0.85)
    selected_slices = data[start:end]

    # Normalize slices to [0,1]
    normalized = [
        np.interp(slice_, (slice_.min(), slice_.max()), (0, 1))
        for slice_ in selected_slices
    ]

    # Resize slices to target_shape
    resized = [
        resize(slice_, target_shape, preserve_range=True, anti_aliasing=True)
        for slice_ in normalized
    ]

    return np.stack(resized)


def load_data(
    directory: str, labels_df: pd.DataFrame,
    target_shape: tuple = (128, 128)) -> tuple:
    """
    Loads MRI data from a directory, extracts middle slices from each MRI volume, 
    and assigns the corresponding labels based on the provided labels dataframe.

    Parameters:
    directory (str): The path to the directory containing the .npz files with MRI volumes.
    labels_df (pd.DataFrame): A dataframe containing the participant IDs and their corresponding 
                              encoded labels (e.g., schizophrenia diagnosis).
    target_shape (tuple): The target height and width for resizing each 2D slice. 
                          Default is (128, 128).

    Returns:
    tuple: A tuple containing two numpy arrays:
           - np.ndarray: A 3D numpy array of the processed MRI slices, 
                         with the shape (num_slices, target_shape[0], target_shape[1]).
           - np.ndarray: A 1D numpy array of the corresponding labels for each slice.
    """

    all_slices = []
    all_labels = []

    npz_files = [f for f in os.listdir(directory) if f.endswith(".npz")]

    for file_name in tqdm(npz_files, desc=f"Loading from {directory}"):
        path = os.path.join(directory, file_name)
        subject_id = extract_subject_number(file_name)
        if not subject_id:
            continue
        label_row = labels_df[labels_df["participant_id"] == subject_id]
        if label_row.empty:
            continue
        label = label_row["dx_encoded"].values[0]

        try:
            # Instead of loading manually, use your own middle-slice function
            slices = load_middle_slices(path, target_shape=target_shape)
            all_slices.extend(slices)
            all_labels.extend([label] * slices.shape[0])
        except Exception as e:
            print(f"Error loading {path}: {e}")

    return np.array(all_slices), np.array(all_labels)


X_train, y_train = load_data(train_path, labels_df)
X_test, y_test = load_data(test_path, labels_df)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
