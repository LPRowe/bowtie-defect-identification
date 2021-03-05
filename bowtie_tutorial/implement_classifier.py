# -*- coding: utf-8 -*-
"""
The final step is to use the trained classifier on new images.  

This script is nearly identical to get_unlabeled_bowties.py

The difference is that before we place a box around a potential bowtie, we extract the 128
features from the region of interest in the shear 0 and shear 45 images, then feed those
features to the machine learning model we trained to predict if it is a bowtie or not.

If it is a bowite, we box it in white, if it is not a bowtie, then we box it in black.

An important item to note here is that the training images do not overlap with the images
used to create the provided training data.

We created a test set and a training set in the last module to test the model on bowties it had not
seen before.  Similarly, if we used images that were in the training set, the model would appear to
be performing unrealistically well.
"""
import pickle 

import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, AnnotationBbox

# Note: suppressing the display of matplotlib here will significantly decrease the runtime

from inspect_the_data import replace_hot_pixels, pixel_to_xy
from reading_the_data import get_images
from get_unlabeled_bowties import subdivide, local_to_global_index 

def extract_features(img0, img45, j, i, xdim=640, ydim=480):
    """
    Gets 128 pixel values from the 8 by 8 box centered at i, j in the shear0 and shear45 images.
    If the pixel is too close to the edge of the image returns a zerofilled array.
    Returns a 1D array with 128 pixel values, flattened img0, then flattened img45
    """
    if (i < 4) or (j < 4) or (i > 476) or (j > 636):
        return np.zeros((1, 128))
    data0 = list(np.reshape(img0[i-4: i+4, j-4: j+4], (64,)))
    data45 = list(np.reshape(img45[i-4: i+4, j-4: j+4], (64,)))
    return np.array(data0 + data45).reshape((1, -1))

def add_boxes(image, points, bowties, size=10, xdim=640, ydim=480):
    """
    Places a white box of edge length 2*size around each point in points.
    Returns the image with boxes added.
    
    image: numpy array of dimensions (xdim, ydim)
    points: a list of points to be boxed
    """
    maxi = np.max(image)
    mini = np.min(image)
    for point, bowtie in zip(points, bowties):
        val = maxi if bowtie else mini # white box for bowties or black box for nonbowties    
        j, i = pixel_to_xy(point)
        for di in range(max(0, i-size), min(ydim, i+size+1)):
            if j + size < xdim:
                image[di][j+size] = val
            if j - size >= 0:
                image[di][j-size] = val
        for dj in range(max(0, j-size), min(xdim, j+size+1)):
            if i + size < ydim:
                image[i+size][dj] = val
            if i - size >= 0:
                image[i-size][dj] = val
    return image
        
def annotate_image(file_name, sub0, sub45, img_number, model, N=5, save_dir='./images/classified_images/', hot_pixels=[15968, 15546], xdim=640, ydim=480):
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
    
    # Classify each location as a bowtie or non bowtie
    features = [extract_features(img0, img45, *pixel_to_xy(i)) for i in peak_pixels]
    bowties = [model.predict(data) for data in features]
    
    # Place a white box around each peak pixel
    img0 = add_boxes(img0, peak_pixels, bowties)
    img45 = add_boxes(img45, peak_pixels, bowties)
    
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
    N = 4 # there will be N * N subregions
    
    # load the previously trained classifier
    model = pickle.load(open('./trained_classifiers/sgd_classifier.pkl', 'rb'))
    
    # Set run all images to true to classify all 25 images
    run_all_images = False
    if not run_all_images:
        # Pick an image from 0 to 24
        IMAGE_NUMBER = 0
        annotate_image(FILES[IMAGE_NUMBER], sub0, sub45, IMAGE_NUMBER, model, N=N)
    else:
        for img_number, file_name in enumerate(FILES):
            print('Annotating',img_number, '/', len(FILES))
            annotate_image(file_name, sub0, sub45, img_number, model, N=N)