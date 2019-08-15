# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 19:32:30 2019

@author: Logan Rowe
"""

import numpy as np
import os
import glob
from myscripts3 import basic_tools as basic
import sys

#UNLOAD MATPLOTLIB (ONLY NEEDED FOR PYTHON ENVIRONMENTS THAT AUTOMATICALLY LOAD MATPLOTLIB ON STARTUP)
modules=[]
for module in sys.modules:
    if module.startswith('matplotlib'):
        modules.append(module)

for module in modules:
    sys.modules.pop(module)
    
#RELOAD WITH 'AGG' TO PREVENT THE SLOW PROCESS OF DISPLAYING EVERY IMAGE ON THE SCREEN
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

from sklearn.externals import joblib

os.chdir('C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification')
import data_prep as dp

import timeit

xdim,ydim=640,480

psdr=False
regen=True

wafer='17' #Wafer to be analyzed
       
#FOLDERS
base_dir=os.getcwd()

savedir=base_dir+'\\Wafer_Images\\'+wafer+'_example_clf_bowties' #save classified bowtie images
subdir=base_dir+'\\Subtraction_Images'                    #subtraction images
hotpixdir=base_dir+'\\Light Levels\\Hot_Pixel_Lists'      #location of hot pixel list
lightleveldir=base_dir+'\\Light_Levels'                   #location of light level data
clf_dir=base_dir+'\\classifiers'





#LOAD SUBIMG
os.chdir(subdir+'\\'+wafer)
sub0=np.load('sub0_low.npy')
sub45=np.load('sub45_low.npy')

#LOAD LIGHT DATA
os.chdir(lightleveldir)
W,AA,AS,SA,SS,MM=np.genfromtxt('WaferLight_AA_AS_SA_SS_MM.txt',delimiter=',',unpack=True).tolist()

try:
    idx=W.index(int(wafer))
    AA,AS,SS,SA=AA[idx],AS[idx],SS[idx],SA[idx]
except:
    pass

#LOAD HOT PIXEL LIST
os.chdir(lightleveldir+'\\Hot_Pixel_Lists')
hotpix=np.genfromtxt(wafer+'_hotpix.txt').tolist()
hotpix=[int(i) for i in hotpix]

datadir=base_dir+'\\Wafer_Data_Files\\'+wafer

os.chdir(datadir)

#LIST OF ALL IMAGES
filenames=[i for i in glob.glob('*.dt1')]

#DESIRED LISTS OF INFORMATION
pf=[] #(pass fail) if image passed light test
imgloc=[] #index of the 5x image on the wafer 0 is the first image taken 3234 is the last
subloc=[] #image sub-divided into smaller images, each sub-image has its own index
peakpixel=[] #the index of the pixel with the greatest shear max intensity within a sub-image
#Note: peakpixel is the pixel index in a sub-image, not in the full image


#ANALYZE ALL IMAGES
count=0
for i in filenames:
    
    #LOAD FILE AND CONVERT TO 3 NUMPY ARRAYS shear 0 image (img0), shear 45 image (img45) and IR-Transmission image (imgL)
    os.chdir(datadir)
    img0,img45,imgL=basic.formimg(i,datadir)
    img0-=sub0
    img45-=sub45
    
    #CHECK TO SEE IF IMAGE PASSES LIGHT LEVEL TEST        
    PF=basic.lightmask(np.mean(imgL),np.std(imgL),AA,SA,AS,SS)
    
    #IF IMAGE FAILS THE LIGHT TEST THEN SKIP THAT IMAGE (image is of wafer mask or blurred)
    if PF==False:
        pf.append(0)
        imgloc.append(count)
        subloc.append(0)
        peakpixel.append(0)
                    
        count+=1
        print('Image '+str(i)+' failed light test')
        continue
    
    #RESET ALL HOT PIXELS TO IMAGE AVERAGE
    val0=np.mean(img0)
    val45=np.mean(img45)
    for j in hotpix:
        img0[j//xdim][j%xdim]=val0 
        img45[j//xdim][j%xdim]=val45
        #image[y][x]=value where y and x are calculated from image width in pixels (xdim) and pixel index (j)
           
    #NOW MAKE SHEAR MAX IMAGE
    imgM=basic.shear_max_img(img0,img45)
    
    
    #IF IMAGE DOES PASS SUBDIVIDE IMAGE AND GET LOCATION OF MAX SHEAR MAX
    dx,dy=160,120 #To make 16 images per 5x shear max image
    dx,dy=128,96
    subimages=basic.subsub(imgM,xdim,ydim,dx,dy)

    #RECORD LOCATIONS OF PEAK RETARDATIONS
    localmaxes=[np.argmax(k) for k in subimages]
    maxidx=[basic.LocalToGlobalIdx(k,m,dx,dy,xdim//dx) for (k,m) in zip(localmaxes,range(0,len(localmaxes)))]
    
    val0_max=np.max(img0)
    val0_min=np.min(img0)
    val45_max=np.max(img45)
    val45_min=np.min(img45)
    

    
    # =============================================================================
    # Check each each peak pixel to see if it belongs to a bowtie
    # using the classifier
    # =============================================================================
    
    clf=joblib.load(clf_dir+'\\XGBC_img_classifier.pkl')
    p_crit=0.8 #yields the highest classifier recall without suffering a serious drop in precision
    
    
    bowtie_classes=[]
    for pixel in maxidx:
        features=dp.extract_features_XGBC(img0,img45,pixel)
        try:
            p=clf.predict_proba(features)
            if p[0][1]>=p_crit:
                bowtie_classes.append(1)
            else:
                bowtie_classes.append(0)
        except:
            bowtie_classes.append(0)
            pass
    
    # =============================================================================
    # Place a white box around each positively identified bowtie
    # =============================================================================
    loccount=0
    for (k,kk) in zip(maxidx,bowtie_classes):        
        pf.append(1)
        imgloc.append(count)
        subloc.append(loccount)
        peakpixel.append(k)
        
        if kk==1:
            #This places a WHITE box around each pixel of interest if it IS classified as a bowtie
            img0=basic.boxpoint(img0,k,val0_max)
            img45=basic.boxpoint(img45,k,val45_max)
        else:
            #This places a BLACK box around each pixel if it is NOT classified as a bowtie
            img0=basic.boxpoint(img0,k,val0_min)
            img45=basic.boxpoint(img45,k,val45_min)

        loccount+=1
        continue
    

    
    '''~~~~~~~~~~~~~ANNOTATE BOXES~~~~~~~~~~~~~~~'''
    # =========================================================================
    # Place the sub-image number above each boxed (non)bowtie and save images
    # =========================================================================
    plt.gray()
    plt.close('all')
    
    fig,ax=plt.subplots()
    ax.imshow(img0)
    
    loccount=0
    for (k,kk) in zip(maxidx,bowtie_classes):  
        offsetbox = TextArea(str(loccount), minimumdescent=False)
        ab = AnnotationBbox(offsetbox,(k%xdim,k//xdim), xybox=(0,25), xycoords='data', boxcoords="offset points")
        ax.add_artist(ab)
               
        loccount+=1
    
    os.chdir(savedir)
    plt.savefig(str(count)+'_0.png')
    
    plt.close('all')
    fig,ax=plt.subplots()
    ax.imshow(img45)
    
    loccount=0
    for k in maxidx:        
        offsetbox = TextArea(str(loccount), minimumdescent=False)
        ab = AnnotationBbox(offsetbox,(k%xdim,k//xdim), xybox=(0,25), xycoords='data', boxcoords="offset points")
        ax.add_artist(ab)
                   
        loccount+=1
    
    os.chdir(savedir)
    plt.savefig(str(count)+'_45.png')
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

    
    count+=1
    
    print(wafer+': '+str(count)+' of '+str(len(filenames)))
    