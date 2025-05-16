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

        # Store only filenames and labels
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

        scan = np.load(scan_path)['data']
        scan = (scan - scan.min()) / (scan.max() - scan.min() + 1e-5)  # Normalize
        scan = torch.tensor(scan, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)

        if self.augment:
            scan = augment_scan(scan)

        return scan, torch.tensor(label, dtype=torch.long)
    
def get_dataloaders(train_dir, val_dir, test_dir, 
                    clinical_data, 
                    batch_train=4, batch_val=2, batch_test=1,
                    num_workers=2):
    # build datasets
    train_dataset = MRI_Dataset_OnTheFly(train_dir, clinical_data, augment=False)
    val_dataset = MRI_Dataset_OnTheFly(val_dir, clinical_data, augment=False)    
    test_dataset  = MRI_Dataset_OnTheFly(test_dir, clinical_data,  augment=False)

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
        batch_size=batch_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader    