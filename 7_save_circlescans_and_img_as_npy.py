# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 00:02:58 2019

@author: Logan Rowe

Creates the numpy array that will be used as the basis of the machine learning training set.

Returns X:
    [[
    wafer int, 
    location of image int, 
    sublocation on image int, 
    pixel at center of bowtie int,
    theta_M the angle of maximum intensity in the shear max image float (radians),
    shear 0 circle sweep around bowtie 72 values floats,
    shear 45 circle sweep around bowtie 72 values floats,
    1 if a bowtie and 0 if nonbowtie
    ],]
    
Synopsis:
    1) Loads shear 0 and shear 45 (non)bowties from .npy files
    2) Calculates shear max image from shear 0 and shear 45 images
    3) Takes the average value of a line scan from the center of the bowtie and
       repeats the process as the line scan is swept in a circular motion around
       the bowtie
    4) From the circle sweep of shear max (step 3) the angle theta_M is measured
       at which the maximum intensity was observed in the shear max image
    5) Circle sweep is done for shear 0 and shear 45 images starting at
       theta_naught=theta_M such that all circle sweeps will start at the angle
       of maximum intensity in the shear max image (this is done to achieve uniformity 
       across bowties with different orientations)
    6) Training data set is built up with the shape (N,150)
       the 150 data points are listed in order under "Returns X:" above
"""

import numpy as np
import os
import sys
import glob
from myscripts3 import basic_tools as basic
import matplotlib.pyplot as plt


bowtie_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowties'
nonbowtie_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\non-bowties'

save_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowtie-training-data'



os.chdir(bowtie_dir)
wafers=glob.glob('*')
print(wafers)


points=72 #how many data points per sweep around a bowtie
bs=8      #Sidelength of box-array around each bowtie

bowtie_data=np.full((0,(2*points)+10),0) #Shape of bowtie data is all of the bowtie line scan points, plus 6 for Wafer, Location, Sublocation, Pixel, theta_M and (non)bowtie identifier
image_data=np.full((0,(bs**2)*2+7),0)    #wafer, location, sublocation, pixel, standard deviation 0, sandard deviation 45, shear0 image (8,8), shear45 image (8x8), (non)bowtie identifier

badbowcount=0
for identity in [('bowties',bowtie_dir,1),('nonbowties',nonbowtie_dir,0)]:
    print(str(badbowcount),'bowties were rejected')
    for wafer in wafers:
        badbowcount=0
        print("Starting",wafer,identity[0])
        os.chdir(identity[1]+'\\'+wafer)
        for (bow0_file,bow45_file) in zip(glob.glob('*0.npy'),glob.glob('*45.npy')):
            if bow45_file.split('_')[-2]=='0':
                #For unknown reason sometimes {wafer}_0_0_0_45.npy somehow makes 
                #its way into the bowtie dataset even though it is not in the data file
                #this will skip the phantom bowtie
                badbowcount+=1
                continue
                        
            bow0,bow45=np.load(bow0_file),np.load(bow45_file)
            
            if bow0.shape[0]!=40 or bow0.shape[1]!=40:
                #Bowties that are truncated by the edge of the image
                #Will not have a clean circle sweep and so we will skip them
                badbowcount+=1
                continue
            
            bowM=basic.shear_max_img(bow0,bow45) #Calculate shear max image from shear 0 and shear 45 images
            
            G=820 #index of central pixel (y_location*xdim+x_location) where y_loc and x_loc are both 20 and xdim is the width of the bowtie array
            R=4 #Length (radius) of line that will be swept around the center of the bowtie [pixels]
            
            meanvals_M,thetas_M=basic.circlesweep(bowM,G,R,res=points,xdim=40)
            
            theta_M=thetas_M[np.argmax(meanvals_M)] #The angle of maximum intensity as measured on the shear max bowtie shear 0 and shear 45 circle scan swill all start from this angle
            
            #Perform circle sweep starting at the angle of maximum shear max so that all bowties are similar in pattern
            meanvals_0,thetas_0=basic.circlesweep(bow0,G,R,res=points,xdim=40,theta_naught=theta_M)
            meanvals_45,thetas_45=basic.circlesweep(bow45,G,R,res=points,xdim=40,theta_naught=theta_M)
            
            bow_loc=[int(loc) for loc in bow0_file.split('_')[:4]]
            bow_vals=[theta_M,thetas_0[np.argmax(meanvals_0)],thetas_45[np.argmax(meanvals_45)],np.std(bow0),np.std(bow45)]
            
            X=[]
            X.extend(bow_loc) #Wafer, location, subloc, pixel
            X.extend(bow_vals) #theta_M, theta0, theta 45, std0, std45
            X.extend(meanvals_0) #circle sweep of shear 0
            X.extend(meanvals_45) #circle sweep of shear 45
            X.append(identity[2]) #bowtie or not (1|0)
            
            bowtie_data=np.append(bowtie_data,[X],axis=0)
            
            ########################################
            # Crop 8x8 images of sh0, sh45 and save
            # second data set
            ########################################
            
            xdim=bow0.shape[1]
            xmin,xmax=int((G//xdim)-(0.5*bs)),int((G//xdim)+(0.5*bs))
            ymin,ymax=int(0.5*(xdim-bs)),int(0.5*(xdim+bs))
            crop0,crop45=bow0[ymin:ymax,xmin:xmax],bow45[ymin:ymax,xmin:xmax]
            
            X2=[]
            X2.extend(bow_loc)
            X2.extend()
            image_data=np.append(image_data,[],axis=0)
            
            
            

'''
Save data as npy file in the format of 
[[ wafer,
location,
sublocation on image,
pixel index of bowtie,
theta_M (rad),
angle of maximum intensity in theta 0 circle sweep,
angle of maximum intensity in theta 45 circle sweep,
standard deviation of pixels in bow0 image,
standard deviation of pixels in bow 45 image,
shear 0 circle sweep, 
shear 45 circle sweep,
bowtie (1) or nonbowtie (0) ]]
'''
os.chdir(save_dir)
np.save('wafer_loc_subloc_pixel_thetaM_theta0_theta45_std0_std45_shear0_shear45_bow-bool.npy',bowtie_data)