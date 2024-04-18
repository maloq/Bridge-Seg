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

#This ideally should be in config file 
BATCH_SIZE = 2
WORKERS = 1
LR = 1e-4
WD = 0.03
MAX_EPOCHS = 60
DINO_MODEL_NAME = 'dinov2_b'

    

class DataModule(pl.LightningDataModule):
    def __init__(self, img_dir, mask_dir, train_images_names=None, val_images_names=None, train_size=50):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.train_images_names = train_images_names
        self.val_images_names = val_images_names
        self.train_size = train_size
        dataset = SegmentationDataset(img_dir=self.img_dir,
                                      mask_dir=self.mask_dir,
                                      num_classes = 1,
                                      img_transform=None,
                                      mask_transform=None)
        assert train_size < len(dataset)
        total_mean, total_std = mean_std_from_dataset(dataset)
        print(f"Dataset mean and std: {total_mean}, {total_std}")

        self.img_transform = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(10),
            v2.RandomPerspective(0.3),
            v2.RandomResizedCrop(scale=(0.8, 1.0), size=(14*64,14*64), antialias=True),  # Or Resize(antialias=True)
            v2.ToTensor(),
            # mean=[102.9221/255, 105.5417/255,  97.7115/255], std=[78.5406/255, 74.1163/255, 71.4406/255]
            v2.Normalize(mean=[total_mean[0]/255, total_mean[1]/255, total_mean[2]/255], std=[total_std[0]/255, total_std[1]/255, total_std[2]/255]),
        ])
        self.mask_transform = v2.Compose([
            v2.Resize((32*8,32*8)),
            v2.ToTensor(),
        ])
        
    def setup(self, stage=None):
        #split dataset randomly with fixed train size
        if (self.train_images_names is None) or (self.val_images_names is None): 
            dataset = SegmentationDataset(
                                        img_dir=self.img_dir,
                                        mask_dir=self.mask_dir,
                                        num_classes = 1,
                                        img_transform=self.img_transform,
                                        mask_transform=self.mask_transform)

            generator = torch.Generator().manual_seed(42) # for reproducibility
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [self.train_size, len(dataset) - self.train_size], generator)    
        #split dataset with fixed images in train and val
        else:
            self.train_dataset = SegmentationDataset(
                                        img_dir=self.img_dir,
                                        mask_dir=self.mask_dir,
                                        num_classes = 1,
                                        img_transform=self.img_transform,
                                        mask_transform=self.mask_transform,
                                        images_names=self.train_images_names)
            
            self.val_dataset = SegmentationDataset(
                                        img_dir=self.img_dir,
                                        mask_dir=self.mask_dir,
                                        num_classes = 1,
                                        img_transform=self.img_transform,
                                        mask_transform=self.mask_transform,
                                        images_names=self.val_images_names) 
        
        print(f'Train dataset len: {len(self.train_dataset)}')
        print(f'Val dataset len: {len(self.val_dataset)}')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True)
    

def prepare_model(checkpoint_path=None):
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
    model = SegmentationNet(backbone=dinov2_backbone,
                            head=conv_head,
                            embedding_size=dino_model_data['embedding_size'],
                            patch_size=dino_model_data['patch_size'])
    if checkpoint_path:
        segmentator = LitSegmentator.load_from_checkpoint(checkpoint_path,
                                                          net=model,
                                                          lr=LR,
                                                          weight_decay=WD)
    else:
        segmentator = LitSegmentator(model, lr=LR, weight_decay=WD)

    return segmentator
    

if  __name__ == '__main__':

    img_dir_resized ='data/BrigeImages/images_6x_downscaled'
    mask_dir_resized ='data/BrigeImages/images_masks_6x_downscaled'
    segmentator = prepare_model()
    dm = DataModule(img_dir_resized, mask_dir_resized, train_images_names=None, val_images_names=None, train_size=50)
    trainer = pl.Trainer(
                        max_epochs=MAX_EPOCHS,
                        precision='16-mixed',
                        accelerator="cuda",
                        check_val_every_n_epoch=3,
)

    trainer.fit(model=segmentator,
                datamodule=dm
                )