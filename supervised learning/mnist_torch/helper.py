
import os 
from directories import *

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imgsave(img, path):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imsave(path, np.transpose(npimg, (1, 2, 0)))

def showgrid(loader) : 
    dataiter = iter(loader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))

#  returns content of a file 
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
#  writes content to a file
def write_file(file_path, content, mode = 'a'):
    with open(file_path, mode) as file:
        file.write(content)

# returns the tree of a directory (as in the tree command)
def list_dir(dir_path, indent=0):
    output_content  = ""
    for item in os.listdir(dir_path):
        output_content += ' ' * indent + item + '\n'
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            list_dir(item_path, indent + 4)
    return output_content
