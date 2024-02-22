from utils import mean_std_from_dataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import torch.nn.functional as F
from create_model import SegmentationNet
from dataset import SegmentationDataset
from torch.hub import load
from create_model import conv_head
from create_model import LitSegmentator
import pytorch_lightning as pl

import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 10
WORKERS = 2
LR = 1e-3
WD = 0.05
MAX_EPOCHS = 60
DINO_MODEL_NAME = 'dinov2_s'

    

def prepare_data(train_images_names=None, val_images_names=None, train_size=50):
   
    dataset = SegmentationDataset(img_dir='/home/teshbek/datasets/BrigeImages/images_6x_downscaled',
                                  mask_dir='/home/teshbek/datasets/BrigeImages/images_masks_6x_downscaled',
                                  num_classes = 1,
                                  img_transform=None,
                                  mask_transform=None)
    assert train_size < len(dataset)
    total_mean, total_std = mean_std_from_dataset(dataset)
    print(f"Dataset mean and std: {total_mean}, {total_std}")

    img_transform = v2.Compose([
        v2.RandomHorizontalFlip(0.5),
        v2.RandomRotation(10),
        v2.RandomPerspective(0.3),
        v2.RandomResizedCrop(scale=(0.8, 1.0), size=(14*64,14*64), antialias=True),  # Or Resize(antialias=True)
        v2.ToTensor(),
        v2.Normalize(mean=[total_mean[0]/255, total_mean[1]/255, total_mean[2]/255], std=[total_std[0]/255, total_std[1]/255, total_std[2]/255]),
    ])

    mask_transform = v2.Compose([
        v2.Resize((32*8,32*8)),
        v2.ToTensor(),
    ])

    #split dataset randomly with fixed train size
    if (train_images_names is None) or (val_images_names is None): 
        dataset = SegmentationDataset(img_dir='/home/teshbek/datasets/BrigeImages/images_6x_downscaled',
                                    mask_dir='/home/teshbek/datasets/BrigeImages/images_masks_6x_downscaled',
                                    num_classes = 1,
                                    img_transform=img_transform,
                                    mask_transform=mask_transform)

        generator = torch.Generator().manual_seed(42) # for reproducibility
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size], generator)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, drop_last=True)
        print(f'Train dataset len: {len(train_dataloader)}')
        print(f'Val dataset len: {len(val_dataloader)}')

    #split dataset with fixed images in train and val
    else:
        train_dataset = SegmentationDataset(
                                    img_dir='/home/teshbek/datasets/BrigeImages/images_6x_downscaled',
                                    mask_dir='/home/teshbek/datasets/BrigeImages/images_masks_6x_downscaled',
                                    num_classes = 1,
                                    img_transform=img_transform,
                                    mask_transform=mask_transform,
                                    images_names=train_images_names)
        
        val_dataset = SegmentationDataset(
                                    img_dir='/home/teshbek/datasets/BrigeImages/images_6x_downscaled',
                                    mask_dir='/home/teshbek/datasets/BrigeImages/images_masks_6x_downscaled',
                                    num_classes = 1,
                                    img_transform=img_transform,
                                    mask_transform=mask_transform,
                                    images_names=val_images_names) 

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, drop_last=True)
        print(f'Train dataset len: {len(train_dataloader)}')
        print(f'Val dataset len: {len(val_dataloader)}')

    return train_dataloader, val_dataloader


def prepare_model():
    dino_models = {
                    'dinov2_s':{
                        'name':'dinov2_vits14_reg',
                        'embedding_size':384,
                        'patch_size':14
                    },
                    'dinov2_b':{
                        'name':'dinov2_vitb14_reg',
                        'embedding_size':768,
                        'patch_size':14
                    },
                    'dinov2_l':{
                        'name':'dinov2_vitl14_reg',
                        'embedding_size':1024,
                        'patch_size':14
                    },
                    'dinov2_g':{
                        'name':'dinov2_vitg14_reg',
                        'embedding_size':1536,
                        'patch_size':14
                    },
                }

    dino_model_data = dino_models[DINO_MODEL_NAME]
    dinov2_backbone = load('facebookresearch/dinov2',  dino_model_data['name'])
    model = SegmentationNet(backbone=dinov2_backbone, head=conv_head, embedding_size=dino_model_data['embedding_size'], patch_size=dino_model_data['patch_size'])
    model = model
    segmentator = LitSegmentator(model, lr=LR, weight_decay=WD)
    return segmentator
    

if  __name__ == '__main__':
   
    segmentator = prepare_model()
    train_dataloader, val_dataloader = prepare_data(train_size=50)
    trainer = pl.Trainer(
                        max_epochs=MAX_EPOCHS,
                        precision='16-mixed',
                        accelerator="cuda",
                        check_val_every_n_epoch=3,
)

    trainer.fit(model=segmentator,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                )