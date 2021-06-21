import torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch_utils  import engine, utils, coco_utils
from torch_utils import transforms  as T
import albumentations as A

import time

PennFudan_dataset_dir = 'dataset/PennFudanPed'
sperm_dataset_root = 'dataset/sperm/images'
sperm_annotations_file = 'dataset/sperm/annotations.json'

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model




def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



def get_albumentations_transforms():

    transforms = A.Compose([
        # A.Normalize(),
        # A.Blur(p=0.5),
        # A.ColorJitter(p=0.5),
        # A.Downscale(p=0.3),
        A.Superpixels(p=0.3),
        A.RandomContrast(p=0.5),
        A.ShiftScaleRotate(p=1),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Sharpen(p = 0.5),

        # A.RGBShift(p=0.5),
        # A.RandomRain(p=0.3),
        # A.RandomFog(p=0.3)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    return transforms