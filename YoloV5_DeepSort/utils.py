



import os
import sys
# sys.path.append(r'C:\Users\gad\Desktop\repos\AI\instance_segmentation')
sys.path.append(r'C:\Users\gad\Desktop\repos\.vscode\Microscope')
import numpy as np
from PIL import Image
import cv2 
import random 

import torch
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.core.composition import BboxParams, Compose
# from pycocotools.coco import COCO
from pycocotools import coco as co

# from my_utils import * 
from python.utils import * 


def my_augmentation():

    transforms = A.Compose([
        # A.Normalize(),
        # A.Blur(p=0.5),
        # A.ColorJitter(p=0.5),
        # A.Downscale(p=0.3),
        # A.Superpixels(p=0.3),
        A.RandomContrast(p=0.5),
        A.ShiftScaleRotate(p=0.8),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Sharpen(p = 0.5),

        # A.RGBShift(p=0.5),
        # A.RandomRain(p=0.3),
        # A.RandomFog(p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    return transforms



def get_yolo_labels(label_path):
    boxes = []
    label_file = open(label_path, 'r')
    for line in label_file:
        _, x_center, y_center, w, h = line.split(' ')
        boxes.append([float(x_center), float(y_center), float(w), float(h)])
    label_file.close()

    return boxes


def write_yolo_labels(label_path, boxes):
    label_file = open(label_path, 'w')
    for box in boxes:
        line = '0 ' + ' '.join([str(i) for i in box]) + '\n'
        label_file.write(line)
    label_file.close()

def denorm_yolo(boxes, image_shape):
    new_boxes = []
    for box in boxes:
        x_center, y_center, w, h = box
        h = int(float(h) * image_shape[0])
        w = int(float(w) * image_shape[1])
        
        y_center = int(float(y_center) * image_shape[0])
        x_center = int(float(x_center) * image_shape[1])

        y = y_center - (h//2)
        x = x_center - (w//2)
        y2 = y + h
        x2 = x + w
        new_boxes.append([x, y, x2, y2])

    return new_boxes


def draw_boxes(image, boxes):

    boxed_image = image.copy()
    for box in boxes:
        x, y, x2, y2 = box
        boxed_image = cv2.rectangle(boxed_image, (int(x), int(y)), (int(x2), int(y2)), (255,0,0), 2)
    return boxed_image
