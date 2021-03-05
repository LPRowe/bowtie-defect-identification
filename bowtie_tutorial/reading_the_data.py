"""
Depending on the source, data will be stored in different formats.  

In this case, the data is from DeltaVision's software and is stored in .dt1 files.
Each file consists of a header with several flags marked with '@**@flag_name'.

The final flag is '@**@data' which is followed by a 307,200 pixel values.
These represent 3 flattened images the Light Image (IR-Transmission), Shear0, and Shear45 images.

This script:
    1. Reads .dt1 files
    2. Parses the data for the 3 images data specifically.
    3. Reshapes the flattened arrays into images.
    4. Displays and / or saves the images.
"""
import array

import numpy as np
import matplotlib.pyplot as plt
import glob

def get_images(filename, xdim = 640, ydim = 480):
    """
    Parses .dt1 file to extract 3 images.
    Returns Light Image, Shear0 Image, Shear45 Image in the form of numpy arrays.
    """
    filestring = open(filename,"rb").read()
    
    # Image data begins 34 bytes after the '@**@Data' flag
    idx = filestring.rfind(b'@**@Data')
    splitstring = filestring[idx+34:]
    f = array.array('f')
    #f.fromstring(splitstring)
    f.frombytes(splitstring)
    f = np.asarray(f)

    # Separate into 3 image types IR Transmission (Light), Shear0, and Shear 45
    imgLight = f[:xdim*ydim].reshape(ydim, xdim)

    # Shear 45 Image (must be normalized by light image)
    img45 = f[xdim*ydim: 2*xdim*ydim].reshape(ydim, xdim)
    with np.errstate(divide='ignore', invalid='ignore'):
        img45 = np.true_divide(img45,imgLight)
        img45[img45 == np.inf] = 1
        img45 = np.nan_to_num(img45)
        
    # Shear 0 Image (must be normalized by light image)
    img0 = f[2*xdim*ydim:3*xdim*ydim].reshape(ydim,xdim)        
    with np.errstate(divide='ignore', invalid='ignore'):
        img0 = np.true_divide(img0,imgLight)
        img0[img0 == np.inf] = 1
        img0 = np.nan_to_num(img0)
    
    return imgLight, img0, img45

def show_images(imgL, img0, img45):
    """Plots Light, Shear0 and Shear45 images in grayscale."""
    plt.gray() 
    plt.figure('Light')
    plt.imshow(imgL)
    plt.title("IR Transmission")
    plt.figure('Shear 0')
    plt.imshow(img0)
    plt.title("Shear 0")
    plt.figure('Shear 45')
    plt.imshow(img45)
    plt.title("Shear 45")
    
def save_images(file_names, save_file = "./images/5x_images/"):
    """Saves a copy of each image."""
    for i, file in enumerate(file_names):
        print('Saving',i,'/',len(file_names))
        imgL, img0, img45 = get_images(file)
        plt.imsave(f'{save_file}{i}_Light.png', imgL)
        plt.imsave(f'{save_file}{i}_Shear0.png', img0)
        plt.imsave(f'{save_file}{i}_Shear45.png', img45)
        
if __name__ == "__main__":
    plt.close('all')
    FILES = glob.glob('./data_files/*.dt1')
    IMAGE_NUMBER = 1 # choose an image to view from 0 to 24
    imgL, img0, img45 = get_images(FILES[IMAGE_NUMBER])
    show_images(imgL, img0, img45)
    save_images(FILES)