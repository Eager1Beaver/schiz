from torch.utils.data import Dataset

from src.preprocess import load_nii, preprocess_image, to_tensor

# TODO: the class is supposed to handle the paths to the data and the labels
# the user provides only a path to the data folder

class MRIDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = load_nii(self.file_paths[idx])
        image = preprocess_image(image)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return to_tensor(image), label