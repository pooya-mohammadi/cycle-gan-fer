from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

import config


class ReferenceTargetDataset(Dataset):
    def __init__(self, root_target, root_reference, transform=None):
        self.root_target = root_target
        self.root_reference = root_reference
        self.transform = transform

        self.target_images = os.listdir(root_target)
        self.reference_images = os.listdir(root_reference)
        self.length_dataset = max(len(self.target_images), len(self.reference_images))  # 1000, 1500
        self.target_len = len(self.target_images)
        self.reference_len = len(self.reference_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        target_img = self.target_images[index % self.target_len]
        reference_img = self.reference_images[index % self.reference_len]

        target_path = os.path.join(self.root_target, target_img)
        reference_path = os.path.join(self.root_reference, reference_img)
        if config.IN_CHANNELS == 3:
            target_img = np.array(Image.open(target_path).convert("RGB"))
            reference_img = np.array(Image.open(reference_path).convert("RGB"))
        else:
            target_img = np.array(Image.open(target_path).convert("L"))
            reference_img = np.array(Image.open(reference_path).convert("L"))
        if self.transform:
            augmentations = self.transform(image=target_img, image0=reference_img)
            target_img = augmentations["image"]
            reference_img = augmentations["image0"]

        return target_img, reference_img
