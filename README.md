# DINOv2 segmentation

Project to implemet semantic segmentation on DINOv2 features


The core of the project is discribed in `main.ipynb` notebook

# How to use 

Images should be in folder `data/BrigeImages/images` 

Annotation file should be in folder `data/BrigeImages/instances_default.json`

## Set up environment

Then install requirements from requirements.txt

`pip install -r requirements.txt`

*requirements list might be incomplete*

## Train
Change constants in `train.py` to try different hyperparameters