import cv2 
from sklearn.metrics import confusion_matrix


def show_image(image):
    cv2.imshow('r', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def read_image(im_path):
    return cv2.imread(im_path) 

def plot_data(data, color): 
    if color is None:
        color = data 
        