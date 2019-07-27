# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 00:02:58 2019

@author: Logan Rowe
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

bowtie_data=np.full((0,(2*points)+5),0) #Shape of bowtie data is all of the bowtie line scan points, plus 5 for Wafer, Location, Sublocation, Pixel and (non)bowtie identifier

for identity in [('bowties',bowtie_dir,1),('nonbowties',nonbowtie_dir,0)]:
    for wafer in wafers:
        print("Starting",wafer,identity[0])
        os.chdir(identity[1]+'\\'+wafer)
        for (bow0_file,bow45_file) in zip(glob.glob('*0.npy'),glob.glob('*45.npy')):
            if bow45_file.split('_')[-2]=='0':
                #For unknown reason sometimes {wafer}_0_0_0_45.npy somehow makes 
                #its way into the bowtie dataset even though it is not in the data file
                #this will skip the phantom bowtie
                continue
                        
            bow0,bow45=np.load(bow0_file),np.load(bow45_file)
            bowM=basic.shear_max_img(bow0,bow45) #Calculate shear max image from shear 0 and shear 45 images
            
            G=820 #index of central pixel (y_location*xdim+x_location) where y_loc and x_loc are both 20 and xdim is the width of the bowtie array
            R=4
            meanvals_M,thetas_M=basic.circlesweep(bowM,G,R,res=points,xdim=40)
            
            theta_M=thetas_M[np.argmax(meanvals_M)]
            
            #Perform circle sweep starting at the angle of maximum shear max so that all bowties are similar in pattern
            meanvals_0,thetas_0=basic.circlesweep(bow0,G,R,res=points,xdim=40,theta_naught=theta_M)
            meanvals_45,thetas_45=basic.circlesweep(bow45,G,R,res=points,xdim=40,theta_naught=theta_M)
            
            bow_loc=[int(loc) for loc in bow0_file.split('_')[:4]]
            
            X=[]
            X.extend(bow_loc)
            X.extend(meanvals_0)
            X.extend(meanvals_45)
            X.append(identity[2])
            
            bowtie_data=np.append(bowtie_data,[X],axis=0)

'''
Save data as npy file in the format of 
[[ wafer,
location,
sublocation on image,
pixel index of bowtie,
shear 0 circle sweep, 
shear 45 circle sweep,
bowtie (1) or nonbowtie (0) ]]
'''
os.chdir(save_dir)
np.save('wafer_loc_subloc_pixel_shear0_shear45_bow-bool.npy',bowtie_data)