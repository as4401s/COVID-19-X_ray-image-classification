"""
@author: asarkar

The function helps with the class imbalance problem.
Images of class with fewer images are oversampled using 
various data augmentations.
"""

import cv2 as cv
import numpy as np
import random

#rotate_images, flip, translation, blur are the helper functions for the 
#apply_augmentation function

def rotate_images(image, scale = 1.0, height = 380, width = 380):
    """
    This function takes an image and rotates them in the range 
    (-25 degrees to +25 degrees)
    
    height = height of image,
    width = width of image
    """
    # finds the center of the image
    center = (height/2,width/2)
    
    # random angle selection between +25 degree to -25 degree
    angle = random.randint(-25,25)
    get_matrix = cv.getRotationMatrix2D(center,angle,scale)
    rotated_image = cv.warpAffine(image,get_matrix,(height,width))
    return rotated_image

def flip (image):
    """
    This function flips the image in right or left.
    """
    
    flipped_image = np.fliplr(image)
    return flipped_image

def translation (image):
    """
    This function takes the image and randomly moves the image 
    in the x and y axis in the scale (-50 to +50)
    """
    
    #random selection of x and y in the range (-50 to 50)
    x = random.randint(-50,50)
    y = random.randint(-50,50)
    rows,cols,z = image.shape
    translate = np.float32([[1,0,x],[0,1,y]])
    translated_image = cv.warpAffine(image,translate,(cols,rows))
    return translated_image

# implementing the augmentations randomly
    
def apply_augmentation (image):
    """
    This function selects a random augmentation and applies them to the image,
    and then returns the image
    """
    #select a random augmentation to apply
    number = random.randint(1,4)
    
    if number == 1:
        image= rotate_images(image, scale =1.0, height=380, width = 380)
            
    if number == 2:
        image= flip(image)
                
    if number ==3:
        image= translation(image)
    
    return image
    