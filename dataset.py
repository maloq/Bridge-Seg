from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes, img_transform=None, mask_transform=None, images_names=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes

        # Only include images for which a mask is found
        if images_names is None:
            self.images_names = [img for img in os.listdir(img_dir) if os.path.isfile(os.path.join(mask_dir, img.split(".")[0] + ".png"))]
        else:
            self.images_names = images_names
        print(f'Dataset len: {len(self.images_names)}')

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        img_name = self.images_names[idx].split(".")[0]
        img_path = os.path.join(self.img_dir, self.images_names[idx])
        mask_path = os.path.join(self.mask_dir, img_name + ".png")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.img_transform:
            image = self.img_transform(image)
        else:
            image = torch.tensor(np.array(image))
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.tensor(np.array(mask))
        mask = mask[0] # mask should be 1 channel
  
        return image, mask