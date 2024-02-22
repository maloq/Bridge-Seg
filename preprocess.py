import cv2
import glob
import os
from tqdm import tqdm
import json
import copy
import numpy as np
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np



def coco_annotations_to_masks(ann_path, images_folder, ext):
    coco = COCO(ann_path)
    new_folder_path = images_folder + '_masks'
    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)
    else:
        print('Masks foldser exists, skipping')
    
    for img_num in tqdm(coco.imgs, desc ="Coco annotations to masks", unit = " images"):
        img = coco.imgs[img_num]
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        coco.showAnns(anns)
        if len(anns)>0:
            mask = coco.annToMask(anns[0])
            for i in range(len(anns)):
                mask += coco.annToMask(anns[i])
        else:
            mask = np.zeros([img['height'], img['width']])
        mask = mask*255
        filename = os.path.join(new_folder_path, img['file_name'].replace(ext, 'png'))
        cv2.imwrite(filename, mask) 

    return new_folder_path


def resize_images_folder(raw_images_folder, downscale_factor, ext, interpolation=cv2.INTER_LINEAR):
    new_folder_path = raw_images_folder + f'_{downscale_factor}x_downscaled'
    os.mkdir(new_folder_path)
    hw_set = set()
    
    for filepath in tqdm(glob.glob(os.path.join(raw_images_folder, f'*.{ext}')), desc ="Images resize and save", unit = " images"):
        image = cv2.imread(filepath)
        h = image.shape[0]
        w = image.shape[1]
        hw_set.add((h, w))
        resized_image = cv2.resize(image, (int(w/downscale_factor), (int(h/downscale_factor))), interpolation=interpolation) 
        filename = os.path.join(new_folder_path, os.path.basename(filepath))
        cv2.imwrite(filename, resized_image) 
    
    if len(hw_set)>1:
        print(f'Not all images are the same size: {hw_set}')

    return new_folder_path

if __name__ == '__main__':
    raw_images_folder = '/home/teshbek/datasets/BrigeImages/images'
    ann_path = '/home/teshbek/datasets/BrigeImages/instances_default.json'
    downscale_factor = 6
    coco_annotations_to_masks(ann_path=ann_path, images_folder=raw_images_folder, ext='JPG')
    resize_images_folder(raw_images_folder, downscale_factor, ext='JPG', interpolation=cv2.INTER_AREA)
    resize_images_folder(raw_images_folder + '_masks', downscale_factor, ext='png', interpolation=cv2.INTER_NEAREST)

