from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from utils.augmentations_on_the_fly import augment_scan
import pandas as pd
import numpy as np
import torch
import os

class MRI_Dataset_OnTheFly(Dataset):
    def __init__(self, scans_dir, csv_path, augment=False):
        self.scans_dir = scans_dir
        self.data_info = pd.read_csv(csv_path)
        self.augment = augment

        # Mapping: participant_id -> label
        self.id_to_label = dict(zip(self.data_info['participant_id'], self.data_info['dx_encoded']))

        # Store only filenames and labels (no full scans!)
        self.samples = []
        for fname in os.listdir(scans_dir):
            if fname.endswith('.npz'):
                participant_id = fname.split('_')[0].replace('sub-', '')
                if participant_id in self.id_to_label:
                    self.samples.append((fname, self.id_to_label[participant_id]))
                else:
                    print(f"Warning: ID {participant_id} not found in CSV, skipping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        scan_path = os.path.join(self.scans_dir, fname)

        scan = np.load(scan_path)['data']  # Assuming npz key is 'data'
        scan = (scan - scan.min()) / (scan.max() - scan.min() + 1e-5)  # Normalize
        scan = torch.tensor(scan, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)

        if self.augment:
            scan = augment_scan(scan)

        return scan, torch.tensor(label, dtype=torch.long)
    
def get_dataloaders(train_dir, test_dir, train_csv, test_csv, batch_size=4):
    # build datasets
    train_dataset = MRI_Dataset_OnTheFly(train_dir, train_csv, augment=False)
    test_dataset  = MRI_Dataset_OnTheFly(test_dir,  test_csv,  augment=False)

    # ---- Build sample-weights for the TRAIN set ----
    labels = [lbl for _, lbl in train_dataset.samples]          # list of 0/1
    class_counts   = torch.bincount(torch.tensor(labels))       # (#healthy, #schiz)
    class_weights  = 1.0 / class_counts.float()                 # inverse freq
    sample_weights = [class_weights[lbl] for lbl in labels]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,          # <-- no shuffle when using sampler
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader    