"""
Most datasets have some form of undesired artifacts.

One such artifact in this dataset is hypersensitive pixels.
A hypersensitive pixel is a pixel that regularly has a value
much higher than the other pixels in the image.  They arise
due to thermal effects in the detector.  

Given their nature, they can be easily identified by recording the
most intense pixel in a sample of images.  

Given the number of pixels in the image, the number of images, 
and assuming all pixels have a roughly equal probability of being
the most intense pixel in the image, we can predict the probability
of any pixel occuring as the most intense pixel N times and replace
those pixels that occur as the most intense pixel too frequently.

Replacement can be done by setting the pixel's value to the mean of it's neighbors.

This process can be done recursively since replacing one hot pixel may reveal another.

Take a look at the images in ./images/5x_images.  Can you spot a hypersensitive pixel?
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import collections
from reading_the_data import get_images, show_images

def find_hot_pixels(file_names):
    """Keeps track of how many times each pixel occurred as the most intense pixel in each image."""
    counter = {'Light': collections.Counter(),
             'Shear0': collections.Counter(),
             'Shear45': collections.Counter()
             }
    for i, file in enumerate(file_names):
        print('Inspecting',i,'/',len(file_names))
        imgL, img0, img45 = get_images(file)
        counter['Light'][np.argmax(imgL)] += 1
        counter['Shear0'][np.argmax(img0)] += 1
        counter['Shear45'][np.argmax(img45)] += 1
    print()
    return counter

def pixel_to_xy(pixel, xdim=640):
    """
    Takes pixel index and returns tuple of (x, y) coordinates. 
    y+ axis is down, x+ axis is right, and (0, 0) is the top left corner.
    """
    return (pixel%xdim, pixel//xdim)

def replace_hot_pixels(image, hot_pixels, xdim=640, ydim=480):
    """Replaces hot pixels with the mean of their 8-directionally adjacent neighbors"""
    for pixel in hot_pixels:
        x, y = pixel_to_xy(pixel)
        neighbors = []
        for i in range(max(0, y-1), min(y+2, ydim)):
            for j in range(max(0, x-1), min(x+2, xdim)):
                if i != j:
                    neighbors.append(image[i][j])
        print()
        print(image[y][x])
        print(neighbors)
        image[y][x] = sum(neighbors) / len(neighbors) if neighbors else 0
    return image
        
if __name__ == "__main__":
    plt.close('all')
    FILES = glob.glob('./data_files/*.dt1')
    
    # Count how many times each pixel occurs as the most intense pixel in an image
    counter = find_hot_pixels(FILES)
    for key in counter:
        print(key)
        print(counter[key])
        print()
        
    print("Suspect:", pixel_to_xy(list(counter['Light'].keys()).pop()))
    print("Suspect:", pixel_to_xy(15546))
    
    # Inspect image before replacing pixel
    imgL, img0, img45 = get_images(FILES[0])
    plt.figure(1)
    plt.imshow(imgL)
    
    
    # Replace the hot pixels
    hot_pixels = [15968, 15546]
    imgL = replace_hot_pixels(imgL, hot_pixels)
    
    # Inspect image after replacing hot pixel (notice the contrast change)
    plt.figure(2)
    plt.imshow(imgL)    