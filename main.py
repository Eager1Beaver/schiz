from src.data_loader import MRIDataset
from src.preprocess_validation import plot_slices
from src.augmentation import get_augmentation

def main():
    data_path = "data"
    dataset  = MRIDataset(data_path)

    # Access first loaded scan and its label
    image, label = dataset[0]
    print(f"Image shape: {image.shape}, Label: {label}")

    # Visual inspection
    plot_slices(image, title=f"Scan labeled as {label}")

if __name__ == '__main__':
    main()