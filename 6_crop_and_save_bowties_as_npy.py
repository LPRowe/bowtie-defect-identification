# -*- coding: utf-8 -*-
"""
Created on Sat Sep 01 15:17:19 2018

@author: Logan Rowe

The manual identification of bowties for wafer 50 have already been provided.
If you desire bowties from other wafers please see the readme.md file for 
information on manually characterizing bowties and saving their locations.

This script takes input of the image location, sub-image location and index of the center
pixel of manually identified bowties and non-bowties.

The image containing each bowtie is loaded, post processed with subtraction image,
and a box of size bs=40 pixels is cropped out around the bowtie and saved as
.npy files for both the shear 0 and shear 45 image.  
"""

import numpy as np
import os
import sys
import glob
from myscripts3 import basic_tools as basic
import matplotlib.pyplot as plt

xdim,ydim=640,480 #dimensions of a typical image in pixels

#DATA FILE WHERE RAW IMAGES ARE STORED
wafer='50'
datafile='D:\\Talbot Runs\\1_'+wafer+'\\dt1_'

#SAVE FILE (WHERE TO SAVE THE NPY FILES OF THE MANUALLY IDENTIFIED BOWTIES AND OTHER)
nonbowsavefile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\non-bowties\\'+str(wafer)
bowsavefile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowties\\'+str(wafer)

for folder in [nonbowsavefile,bowsavefile]:
    if os.path.exists(folder)!=True:
        os.makedirs(folder)

os.chdir(datafile)
images=glob.glob('*')

#LOAD SUBTRACTION IMAGE
subfile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Subtraction_Images\\'+wafer

os.chdir(subfile)
sub0=np.load('sub0_low.npy')
sub45=np.load('sub45_low.npy')

#LOAD LOCATIONS OF MANUALLY IDENTIFIED BOWTIES AND OTHER
bowtiefile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images'
os.chdir(bowtiefile)
bowties=np.genfromtxt(wafer+'_imgloc_subloc manual ID bowties.txt',delimiter=',')
other=np.genfromtxt(wafer+'_imgloc_subloc manual ID other.txt',delimiter=',')

#LOAD PIXEL ASSOCIATED WITH EACH BOWTIE
pf,L,S,P=np.genfromtxt(wafer+'_pf_imgloc_subloc_peakpixel.txt',delimiter=',',unpack=True).tolist() #pass fail light test, image location, subimg location, pixel


for identity in [(bowties,bowsavefile,'Bowties'),(other,nonbowsavefile,'Nonbowties')]:
    print('Starting',identity[2])
    for loc in identity[0]:
        imgloc=int(loc[0]) #image bowtie is on
        subloc=int(loc[1]) #subimage number of bowtie
        pixel=int(P[L.index(imgloc)+subloc]) #Loads the pixel location of the bowtie
        
        X=pixel%xdim  #Xcoordinate of center of bowtie in image
        Y=pixel//xdim #Y coordinate of center of bowtie in image
        
        #LOAD IMAGE
        img0,img45,imgL=basic.formimg(images[imgloc],datafile)
        
        #APPLY SUBTRACTION IMAGE
        img0-=sub0
        img45-=sub45
        
        #CROP OUT BOWTIE
        bs=40 #HOW LARGE OF A BOX AROUND EACH BOWTIE IN PIXELS
        
        bow0=img0[int(Y-0.5*bs):int(Y+0.5*bs),int(X-0.5*bs):int(X+0.5*bs)]
        bow45=img45[int(Y-0.5*bs):int(Y+0.5*bs),int(X-0.5*bs):int(X+0.5*bs)]
        
        os.chdir(identity[1])
        np.save(wafer+'_'+str(imgloc)+'_'+str(subloc)+'_'+str(pixel)+'_0.npy',bow0)
        np.save(wafer+'_'+str(imgloc)+'_'+str(subloc)+'_'+str(pixel)+'_45.npy',bow45)
        
        """
        #PLOTTING FOR TESTING PURPOSES
        plt.gray()
        plt.close('all')
        
        plt.figure("shear 0")
        plt.imshow(img0)
        
        plt.figure("shear 45")
        plt.imshow(img45)
        
        plt.figure(1)
        plt.imshow(bow0)
        
        plt.figure(2)
        plt.imshow(bow45)
        """
        
        print(loc)
