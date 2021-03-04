import numpy as np
import matplotlib.pyplot as plt
import glob
from reading_the_data import get_images, show_images

def create_subtraction_images(file_names, xdim=640, ydim=480):
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
    FILES = glob.glob('./data_files/*.dt1')
    
    # Before Images
    imgL, img0, img45 = get_images(FILES[1])
    plt.figure(0)
    plt.imshow(img45)
    plt.title('Shear45')
    
    # Create subtractoin images
    sub0, sub45 = create_subtraction_images(FILES)
    plt.figure(1)
    plt.imshow(sub45)
    plt.title('Sub45')
    
    # Apply subtraction images
    imgL, img0, img45 = get_images(FILES[1])
    img0 -= sub0
    img45 -= sub45
    plt.figure(2)
    plt.imshow(img45)
    plt.title('Shear45 - Sub45')
