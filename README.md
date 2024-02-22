# DINOv2 segmentation

Project to implemet semantic segmentation on DINOv2 features


The core of the project is discribed in `main.ipynb` notebook

# How to use 

Images should be in folder `data/BrigeImages/images` 

Annotation file should be in folder `data/BrigeImages/instances_default.json`

## Set up environment
Clone DINOv2 repo:

`git clone https://github.com/facebookresearch/dinov2`

And install according to repo readme(not extended version)
Then install additional requirements from requirements.txt. Better to use new conda environment

`pip install -r requirements.txt`

*requirements list might be incomplete*

## Train
Change constants in `train.py` to try different hyperparameters