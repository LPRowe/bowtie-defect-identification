"""
Imperfections in the light source, optic path, and detector can result in optical aberrations.

When an aberration is consistently present in the same location, across all images, while the other 
features on the image change, then a subtraction image can be used to remove the optical aberration.

Given the limited data in this tutorial, we will use all of the provided images to create
a subtraction image.  

We will create one subtraction image for the shear 0 images, and one for the shear 45 images.

This is done by simply averaging all of the images.  
sub0[i][j] = sum(img0[i][j] for img0 in all_shear0_images) / number_of_shear0_images

Then as the name suggests, we can subtract the subtraction image from the raw image.

Caution: Subtraction images are not perfect.  They can be used to remove consistent optical aberrations
from an image, but they also affect the remaining pixel values.  Consider what additional information
in the image may be lost or altered when applying a subtraction image.  
"""
import numpy as np
import matplotlib.pyplot as plt
import glob

from reading_the_data import get_images, show_images

def create_subtraction_images(file_names, xdim=640, ydim=480):
    """Creates 2 subtraction images by averaging all shear0 and shear45 images."""
    sub0 = np.zeros((ydim, xdim))
    sub45 = np.zeros((ydim, xdim))
    for i, file_name in enumerate(file_names):
        print(i,'/',len(file_names))
        imgL, img0, img45 = get_images(file_name)
        sub0 += img0
        sub45 += img45
    return sub0 / len(file_names), sub45 / len(file_names)

if __name__ == "__main__":
    plt.close('all')
    FILES = glob.glob('./data_files/*.dt1') # get a list of the path to each .dt1 file
    
    # Pick an image between 0 and 24
    IMAGE_NUMBER = 20
    
    # Show raw Images
    imgL, img0, img45 = get_images(FILES[IMAGE_NUMBER])
    plt.figure(0)
    plt.imshow(img45)
    plt.title('Shear45')
    
    # Create subtraction images
    sub0, sub45 = create_subtraction_images(FILES)
    
    # Save subtraction images as .npy so they can be easily loaded as a numpy array later
    np.save('./images/subtraction_images/sub0.npy', sub0)
    np.save('./images/subtraction_images/sub45.npy', sub45)
    
    # Show subtraction images
    plt.figure(1)
    plt.imshow(sub45)
    plt.title('Sub45')
    
    # Apply subtraction images
    imgL, img0, img45 = get_images(FILES[IMAGE_NUMBER])
    img0 -= sub0
    img45 -= sub45
    plt.figure(2)
    plt.imshow(img45)
    plt.title('Shear45 - Sub45')