import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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

