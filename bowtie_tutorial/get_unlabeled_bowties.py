# -*- coding: utf-8 -*-
"""
Now that we know how to remove optical aberations and hypersensitive pixels,
it's time to get the training data.  

This is often a very time intensive step, in particular for supervised learning
models that require labeled data.

In this script we will identify 25 regions per image that can be used for training data.

The image will be split into 25 evenly sized regions and in each region a box is placed
around the most intense pixel.  The intensity of pixel (i, j) is img0[i][j]**2 + img45[i][j]**2.

The most intense pixel is chosen in each region, because bowties tend to contain the highest
intensity pixel in a subregion.  

The end result will be 25 Shear0 and Shear45 images that each have 25 boxed regions.
The boxed regions will be characterized by a human as either containing
a bowtie or not containing a bowtie.  
"""
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, AnnotationBbox

from inspect_the_data import replace_hot_pixels, pixel_to_xy
from reading_the_data import get_images

def subdivide(image, N, xdim=640, ydim=480):
    """
    Divides image into N*N subimages.
    Returns a list of these subimages.
    """
    dx, dy = xdim // N, ydim // N
    subimages = []
    for i in range(0, ydim, dy):
        for j in range(0, xdim, dx):
            subimages.append(image[i:i+dy, j:j+dx])
    return subimages

def local_to_global_index(i,j,dx,dy,N):
    """Converts the index of peak value in a subdivided image to the index in the original image."""
    return N*(j//N)*dx*dy + N*dx*(i//dx) + dx*(j%N) + i%dx

def add_boxes(image, points, size=10, xdim=640, ydim=480):
    """
    Places a white box of edge length 2*size around each point in points.
    Returns the image with boxes added.
    
    image: numpy array of dimensions (xdim, ydim)
    points: a list of points to be boxed
    """
    max_val = np.max(image)
    for point in points:
        j, i = pixel_to_xy(point)
        for di in range(max(0, i-size), min(ydim, i+size+1)):
            if j + size < xdim:
                image[di][j+size] = max_val
            if j - size >= 0:
                image[di][j-size] = max_val
        for dj in range(max(0, j-size), min(xdim, j+size+1)):
            if i + size < ydim:
                image[i+size][dj] = max_val
            if i - size >= 0:
                image[i-size][dj] = max_val
    return image
        
def annotate_image(file_name, sub0, sub45, img_number, N=4, save_dir='./images/annotated_images/', hot_pixels=[15968, 15546], xdim=640, ydim=480):
    """
    Annotate the image by placing a box around potential bowties.
    Applies image processing steps: subtraction image and hot pixel removal
    Divides the image into N*N subimages.
    For each subimage, places a box around the most intense pixel.
    The intensity of pixel (i, j) is img0[i][j]**2 + img45[i][j]**2.
    Saves the annotated image.
    
    file_name: path to .dt1 image file
    sub0, sub45: subtraction images (numpy array)
    img_number: number of the image to be saved
    save_dir: location where the annotated images should be saved
    hot_pixels: the pixel id of known hypersensitive pixels
    xdim, ydim: dimensions of the image in pixels (typically 640 by 480)
    """
    # Load the IR transmission, shear0, and shear 45 images
    imgL, img0, img45 = get_images(file_name)
    
    # Replace the hot pixels
    img0 = replace_hot_pixels(img0, hot_pixels)
    img45 = replace_hot_pixels(img45, hot_pixels)
    
    # Apply the subtraction image
    img0 -= sub0
    img45 -= sub45
    
    # Split the shear max image into smaller images
    imgM = img0**2 + img45**2
    small_images = subdivide(imgM, N)
    
    # Record the maximum pixel in each small image
    peak_pixels = [np.argmax(img) for img in small_images]
    peak_pixels = [local_to_global_index(i, j, xdim//N, ydim//N, N) for j,i in enumerate(peak_pixels)]
    
    # Place a white box around each peak pixel
    img0 = add_boxes(img0, peak_pixels)
    img45 = add_boxes(img45, peak_pixels)
    
    # Annotate each box with a number
    for image, name in [(img0, str(img_number) + '_0'), (img45, str(img_number) + '_45')]:
        plt.close('all')
        plt.gray()
        fig,ax=plt.subplots()
        ax.imshow(image)
        for j,i in enumerate(peak_pixels):        
            offsetbox = TextArea(str(j), minimumdescent=False)
            box = AnnotationBbox(offsetbox, (i%xdim, i//xdim), xybox=(0, 25), 
                                 xycoords='data', boxcoords="offset points")
            ax.add_artist(box)
        
        # Save the annotated image
        plt.savefig(f"{save_dir}{name}.png")

if __name__ == "__main__":
    
    # Get subtraction images, hot pixels, and file names
    sub0 = np.load('./images/subtraction_images/sub0.npy')
    sub45 = np.load('./images/subtraction_images/sub45.npy')
    hot_pixels = [15968, 15546]
    FILES = glob.glob('./data_files/*.dt1')
    
    # Choose how many subregions to split the image into 
    N = 5 # there will be 5 * 5 = 25 subregions
    
    run_all_images = False
    if not run_all_images:
        # Pick an image from 0 to 24
        IMAGE_NUMBER = 0
        annotate_image(FILES[IMAGE_NUMBER], sub0, sub45, IMAGE_NUMBER, N=N)
    else:
        for img_number, file_name in enumerate(FILES, 1):
            print('Annotating',img_number, '/', len(FILES))
            annotate_image(file_name, sub0, sub45, img_number)