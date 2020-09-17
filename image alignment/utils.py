

import cv2 
import os 
import numpy as np
from  skimage.feature import canny

class filters:

    gaussian_filter = (1.0 / 140.0) * np.array([[1, 1, 2, 2, 2, 1, 1],
                                                [1, 2, 2, 4, 2, 2, 1],
                                                [2, 2, 4, 8, 4, 2, 2],
                                                [2, 4, 8, 16, 8, 4, 2],
                                                [2, 2, 4, 8, 4, 2, 2],
                                                [1, 2, 2, 4, 2, 2, 1],
                                                [1, 1, 2, 2, 2, 1, 1]])

    grad_x_filter = (1.0 / 3.0) * np.array([[-1, 0, 1],
                                        [-1, 0, 1],
                                        [-1, 0, 1]])

    grad_y_filter = (1.0 / 3.0) * np.array([[1, 1, 1],
                                        [0, 0, 0],
                                        [-1, -1, -1]])

def convolve(image, filter):
    k = filter.shape[0] //2 
    new_image = np.pad(image, (k,), 'edge') 
    result = np.zeros(image.shape) 

    def mul(p1, p2):
        return np.sum(np.multiply(p1, p2)) 

    for row in range(k, new_image.shape[0]-k):
        for col in range(k, new_image.shape[1]-k):
            result[row-k,col-k] = mul(new_image[row-k:row+k+1 , col-k:col+k+1], filter)
    return result.astype(np.uint8) 

def grad_mag_angle(gradx, grady):
    mag = np.sqrt(gradx**2 + grady**2) 
    angle = np.arctan(grady/gradx) 
    return mag.astype(np.uint8), angle

def load_image(path):
    return cv2.imread(path) 

def toGray(image): 
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

def show_image(image):
    cv2.imshow('r', image) 
    cv2.waitKey(0) 
    cv2.destroyWindow('r') 

def localMaximization(gradientData, gradientAngle):
    """
    Applies Non-Maxima suppression to gradient magnitude image
    :param gradientData: gradient image
    :param gradientAngle: gradient angle
    :param height:
    :param width:
    :return: the gradient magnitude image after non-maxima suppression
    """
    gradient = np.empty(gradientData.shape)
    numberOfPixels = np.zeros(shape=(256))
    edgePixels = 0

    for row in range(5, gradientData.shape[0] - 5):
        for col in range(5, gradientData.shape[1] - 5):
            theta = gradientAngle[row, col]
            gradientAtPixel = gradientData[row, col]
            value = 0

            # Sector - 1
            if (0 <= theta <= 22.5 or 157.5 < theta <= 202.5 or 337.5 < theta <= 360):
                if gradientAtPixel > gradientData[row, col + 1] and gradientAtPixel > gradientData[row, col - 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            # Sector - 2
            elif (22.5 < theta <= 67.5 or 202.5 < theta <= 247.5):
                if gradientAtPixel > gradientData[row + 1, col - 1] and gradientAtPixel > gradientData[
                    row - 1, col + 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            # Sector - 3
            elif (67.5 < theta <= 112.5 or 247.5 < theta <= 292.5):
                if gradientAtPixel > gradientData[row + 1, col] and gradientAtPixel > gradientData[row - 1, col]:
                    value = gradientAtPixel
                else:
                    value = 0

            # Sector - 4
            elif 112.5 < theta <= 157.5 or 292.5 < theta <= 337.5:
                if gradientAtPixel > gradientData[row + 1, col + 1] \
                        and gradientAtPixel > gradientData[row - 1, col - 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            gradient[row, col] = value

            # If value is greater than one after non maxima suppression
            if value > 0:
                edgePixels += 1
                try:
                    numberOfPixels[int(value)] += 1
                except:
                    print('Out of range gray level value', value)

    print('Number of Edge pixels:', edgePixels)
    return gradient.astype(np.uint8), numberOfPixels, edgePixels

def pTile(percent, imageData, numberOfPixels, edgePixels, file):
    """
    Applies p-tile method of automatic thresholding to find the best threshold value and then apply that
    to the image to create a binary image.
    :param percent: of non zero pixels to be over the threshold
    :param imageData: input image array
    :param numberOfPixels: counts total number of pixels in the image
    :param edgePixels: counts pixels present at the edges
    :param file:
    :return: binary image array with p-tile method thresholding applied
    """
    # Number of pixels to keep
    threshold = np.around(edgePixels * percent / 100)
    sum, value = 0, 255
    for value in range(255, 0, -1):
        sum += numberOfPixels[value]
        if sum >= threshold:
            break

    for i in range(imageData.shape[0]):
        for j in range(imageData[i].size):
            if imageData[i, j] < value:
                imageData[i, j] = 0
            else:
                imageData[i, j] = 255

    return imageData



def Canny_detector(image):
    gray = toGray(image) 
    blurred = convolve(gray, filters.gaussian_filter) 
    
    grad_x = convolve(blurred, filters.grad_x_filter) 
    grad_y = convolve(blurred, filters.grad_y_filter) 
    
    mag, angle = grad_mag_angle(grad_x, grad_y)  
    show_image(mag) 

    gradient, numberOfPixels, edgePixels = localMaximization(mag, angle)  
    show_image(gradient) 
    
    final = pTile(40, gradient, numberOfPixels, edgePixels, gray) 
    return final