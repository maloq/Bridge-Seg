# Bridge Segmentation

The results of the project are described in the `main.ipynb` notebook.

# How to use:

- `mesh_segmentation.ipynb` - notebook for creating masks from a 3D mesh
  - `mesh_to_mask.py` - Python code for functions used in the notebook
- `mmseg_train.ipynb` - notebook for training and evaluating a semantic segmentation model based on DeepLabV3Plus using the mmsegmentation framework, including data preprocessing
- `dinov2_finetune.ipynb` - notebook for training and evaluating a semantic segmentation model based on dinov2 features (not used)
  - `dataset.py` - lightning data module for dinov2 model training
  - `create_model.py` - torch model architecture for dinov2 model training
  - `train.py` - training script for dinov2 model training
- `preprocess.py` - Python code for resizing images and creating masks from a COCO labels file
- `utils.py` - utility functions

## Folders:
- `configs` - contains the mmsegmentation config files used
- `weights` - contains the trained weights
- `data` - expected to contain training masks and images. The path to the training data should be specified in the training scripts.

## Set up the environment:

Create a new environment and install the required packages:

```
conda create --name openmmlab python=3.10 -y
conda activate openmmlab
pip install -r requirements.txt
```

Then, install mmsegmentation and its dependencies:

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
```

For more information, refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).
