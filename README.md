# Brige segmentation

The results of the project is decribed ib `main.ipynb` notebook

# How to use 


* `mesh_segmentation.ipynb` - notebook with creation of masks from 3D mesh 
* * `mesh_to_mask.py` - code for functions used in notebook
* `mmseg_train.ipynb` - notebook with training and evaluation of semantic segmentation model based on DeepLabV3Plus, using mmsegmentatiion framework. Includes data preprocessing 
* `dinov2_finetune.ipynb` - notebook with training and evaluation of semantic segmentation model based on dinov2 features. Includes data preprocessing. Not used. 
* * `dataset.py` - lightning data module for dinov2 model training
* *  `create_model.py` - torch model architecture for dinov2 model training
* * `train.py` - training script for dinov2 model training
* `preprocess.py` - code for images resizing and creation of masks from COCO labels file
* `utils.py` - some utity functions

* Folder `configs` contains used mmsegmentation config files
* Folder `weights` contains trained weights 
* Folder `data` expected to contain training masks and images. Path to training data should be specified in training scripts.

## Set up environment

Create env and install from requirements.txt:

```
conda create --name openmmlab python=3.10 -y
conda activate openmmlab
pip install -r requirements.txt
```
then install install mmsegmentation and dependencies
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"

```
More about [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) installation
