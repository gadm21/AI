import os

from dataset import COCODataset
from transform import *


data_path = 'data' # os.path.join(parentdir, 'data')
sperm_dataset_path = os.path.join(data_path, 'sperm_dataset')
sperm_images_path = os.path.join(sperm_dataset_path, 'images')
annotations_file_path = os.path.join(sperm_dataset_path, 'annotations.json')



def main():
    print("ann file:", annotations_file_path)
    print("root:", sperm_images_path)
    albumentations_transforms = get_albumentations_transforms( mode = 'train')
    dataset = COCODataset(root_dir = sperm_images_path, coco_path=annotations_file_path, transforms = albumentations_transforms)

    image, target = dataset[0]

    print(type(image))
    print(type(target))
    print(target.keys())

if __name__ == '__main__':
    main()