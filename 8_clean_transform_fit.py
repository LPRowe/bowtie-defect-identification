# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 22:56:03 2019

@author: Logan Rowe
"""

import numpy as np
import os
import sys

datafile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowtie-training-data'
X_=np.load(datafile+'\\wafer_loc_subloc_pixel_shear0_shear45_bow-bool.npy')

'''
BALANCING THE DATA SET

There are currently W: (B,N) bowties&non-bowties identified for each wafer:
17: (192,513)
27: (189,512)
50: (472,1118)
93: (110,507)

To balance the data set we will take an equal number of bowties and non-bowties from each wafer.
'''
##################################################################
#  Create a separate array for each wafer's data
##################################################################

for W in set(X_[:,0]):
    print(W)
    globals()['X_%s'%str(int(W))]=X_[(X_[:,0]==W),:] 
    
##################################################################
#  Rebuild dataset with a balanced number of bowties and non-bowties
##################################################################
X=np.full((0,X_.shape[1]),0)

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

##################################################################
#  Remove columns that will not be needed for training the 
#  machine learning classifier
##################################################################
X=X[:,5:] #Keep only shear 0, shear 45 , and bowtie identifier
            
            
##################################################################
#  Reduce the dimensionality of X by a factor of:
#  2,3,4,6,8, or 9
##################################################################    
reduce=3   
mask=[bool((i)%reduce) for i in range(X.shape[1]-1)]
mask=np.logical_not(mask).tolist()
mask.append(True) #Always keep the bowtie identifier

X=X[:,mask]
            
            
            
            
            
            
            
            
            
            
            
            