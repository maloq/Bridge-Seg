import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def mean_std_from_dataset(dataset):

    image_loader = DataLoader(dataset, 
                            batch_size=50, 
                            shuffle=False, 
                            num_workers=0,
                            pin_memory=True)
    means = torch.tensor([0,0,0]).float()
    stds = torch.tensor([0,0,0]).float()
        
    for batch in tqdm(image_loader):
        image = batch[0]
        image = torch.moveaxis(image, -1, 1)
        means += image.float().mean(axis = [0, 2, 3])
        stds += image.float().std(axis = [0, 2, 3])
        
    total_mean = means/len(image_loader)
    total_std = stds/len(image_loader)
    return total_mean, total_std


def overlay(image, mask, color, alpha, resize=None):

    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


