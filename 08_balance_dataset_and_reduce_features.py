# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 22:56:03 2019

@author: Logan Rowe

NOTE:
    Two data sets are being pepared for ML training using different features.
    The purpose of doing so is to train different classifiers that will make
    different mistakes and benefit from ensemble learning.

BALANCING THE DATA SET

There are currently W: (B,N) bowties&non-bowties identified for each wafer:
17: (192,513)
27: (189,512)
50: (472,1118)
93: (110,507)

To balance the data set we will take an equal number of bowties and non-bowties from each wafer.
This will achieve:
17: (192,192)
27: (189,189)
50: (472,472)
93: (110,110)
    (963,963) <-- Total
    
FEATURE REDUCTION FOR ALL DATASETS

Furthermore the dataset contains N features and 1 classification, this script 
will remove the features that do not aid in bowtie identification: 
    
    Remove:
        Wafer
        Location
        Sublocation
        Pixel
        Theta_M (although this variable is worth considering)
        
OPTIONAL FEATURE REDUCTION FOR BOWTIE CIRCLE SWEEP DATASET ONLY:

Finally we will reduce the resolution of the circle sweep by a variable factor
of 1, 2, 3, 4, 6, 8, or 9 and thus also reduce the total dimensions.

The default is reduce=1 because circle sweep feature reduction is offered
as a hyperparameter in script 9 and thus does not need to be reduced here


"""

import numpy as np
import os
import sys

datafile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowtie-training-data'

#Circle Sweep Data
X_=np.load(datafile+'\\wafer_loc_subloc_pixel_thetaM_theta0_theta45_std0_std45_shear0_shear45_bow-bool.npy')
print(X_.shape)

#Cropped Shear 0 and Shear 45 Image Arrays
X2_=np.load(datafile+'\\wafer_loc_subloc_pixel_std0_std45_shear0_shear45_bow-bool.npy')
print(X2_.shape)

##################################################################
#  Create a separate array for each wafer's data
##################################################################

#Circle Sweep Data
for W in set(X_[:,0]):
    print('Making circle sweep array for wafer',str(W))
    globals()['X_%s'%str(int(W))]=X_[(X_[:,0]==W),:] 
    
#Cropped Img Data
for W in set(X2_[:,0]):
    print('Making image array for wafer',str(W))
    globals()['X2_%s'%str(int(W))]=X2_[(X2_[:,0]==W),:]   
    
##################################################################
#  Rebuild dataset with a balanced number of bowties and non-bowties
##################################################################
X=np.full((0,X_.shape[1]),0)
X2=np.full((0,X2_.shape[1]),0)

#Circle Sweep Data
for W in set(X_[:,0]):
    non_bow_count=0 #keep track of how many non-bowties are allowed into X
    total_bowties=int(np.sum(globals()['X_%s'%str(int(W))][:,-1])) #Total number of bowties from Wafer W
    
    for row in globals()['X_%s'%str(int(W))]:
        if row[-1]==1:
            X=np.append(X,[row],axis=0)
        else:
            if non_bow_count<total_bowties:
                X=np.append(X,[row],axis=0)
                non_bow_count+=1
            else:
                continue

#Cropped Img Data
for W in set(X2_[:,0]):
    non_bow_count=0 #keep track of how many non-bowties are allowed into X
    total_bowties=int(np.sum(globals()['X2_%s'%str(int(W))][:,-1])) #Total number of bowties from Wafer W
    
    for row in globals()['X2_%s'%str(int(W))]:
        if row[-1]==1:
            X2=np.append(X2,[row],axis=0)
        else:
            if non_bow_count<total_bowties:
                X2=np.append(X2,[row],axis=0)
                non_bow_count+=1
            else:
                continue

##################################################################
#  Remove columns that will not be needed for training the 
#  machine learning classifier
##################################################################
                
#Circle Sweep Data
X=X[:,4:] #Remove data related to wafer, location, sublocation and pixel index  

#Cropped Img Data
X2=X2[:,4:]           
            
##################################################################
#
#  Reduce the dimensionality of X by a factor of:
#  1, 2, 3, 4, 6, 8, or 9
#
#  Only done for Circle Sweep Data, NOT FOR CROPPED IMG DATA
#
##################################################################    
reduce=1
mask=[bool((i)%reduce) for i in range(X.shape[1]-6)] #Masks every 3rd (or reduce value) data point in the shear 0 and shear 45 sweeps
mask=np.logical_not(mask).tolist()
mask.append(True) #Always keep the bowtie identifier
mask_thetas_stds=[True]*5
mask=mask_thetas_stds+mask

#Circle Sweep Data
X=X[:,mask]
np.save(datafile+'\\thetaM_theta0_theta45_std0_std45_sh0_sh45_bow-bool_train_'+str(int(144/reduce))+'_dim.npy',X)           

np.save(datafile+'\\std0_std45_sh0-arr_sh45-arr_bow-bool_train.npy',X2)           