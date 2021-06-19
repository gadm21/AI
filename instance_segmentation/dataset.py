import os
import numpy as np
import torch
from PIL import Image
import cv2 

# from pycocotools.coco import COCO
from pycocotools import coco as co

class PennFudanDataset(object):
    def __init__(self, root, transforms = None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)








class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = co.COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # masks
        w, h = img.size
        masks = np.zeros((num_objs, h, w))

        for obj in range(num_objs):
            masks[obj, :, :] = self.coco.annToMask(coco_annotation[obj])
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation['masks'] = masks
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)



def show_image(image):
    cv2.imshow('mask', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test():

    dataset = PennFudanDataset('dataset/PennFudanPed')
    image, target = dataset[0]
    print(target.keys())
    masks = target['masks']
    for i in range(masks.shape[0]):
        mask = masks[i]
        print(np.unique(mask))
        mask = np.array(mask*255, dtype = np.uint8)
        show_image(mask)

    return


    # images_path = r'dataset\sperm\images'
    # annotations = r'dataset\sperm\annotations.json'
    # coco = co.COCO(annotations)

    # # cat_ids = coco.getCatIds()
    # img_ids = coco.getImgIds()
    # anns_ids = coco.getAnnIds(imgIds = 9)
    # anns = coco.loadAnns(anns_ids)

    # image = cv2.imread(os.path.join(images_path, coco.imgs[0]['file_name']))
    # h, w = image.shape[:2]
    # mask = np.zeros((h, w))
    # ann_id = 1
    # for ann in anns:
    #     mask = np.maximum(mask, coco.annToMask(ann)*ann['category_id']*ann_id)
    #     ann_id += 1
    # # mask = np.array((mask/mask.max())*255, dtype = np.uint8)
    # objects = np.unique(mask)
    # objects = objects[1:] # remove the background label
    # print(mask.shape, '  ', objects.shape)
    # masks = mask == objects[:, None, None]
    # print(masks.shape)
    # for i in range(masks.shape[0]):
    #     mask = masks[i, :, :]
    #     mask = np.array((mask/mask.max())* 255, dtype = np.uint8)

    return





def main():

    root = r'dataset\sperm\images'
    annotations = r'dataset\sperm\annotations.json'
    coco = myOwnDataset(root, annotations)


if __name__ == "__main__":

    main()
    