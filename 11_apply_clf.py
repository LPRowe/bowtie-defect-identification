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
import matplotlib.pyplot as plt

from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

from sklearn.externals import joblib

import data_prep as dp

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
clf_dir=basedir+'\\classifiers'





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
    
    os.chdir(datadir)
    img0,img45,imgL=basic.formimg(i,datadir)
    img0-=sub0
    img45-=sub45
    
    #CHECK TO SEE IF IMAGE PASSES LIGHT LEVEL TEST        
    PF=basic.lightmask(np.mean(imgL),np.std(imgL),AA,SA,AS,SS)
    
    #IF IMAGE FAILS THE LIGHT TEST THEN SKIP THAT IMAGE (CONTINUE)
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
    subimages=basic.subsub(imgM,xdim,ydim,dx,dy)

    #RECORD LOCATIONS OF PEAK RETARDATIONS
    localmaxes=[np.argmax(k) for k in subimages]
    maxidx=[basic.LocalToGlobalIdx(k,m,dx,dy,xdim//dx) for (k,m) in zip(localmaxes,range(0,len(localmaxes)))]
    
    val=np.max(imgM)
    

    
    # =============================================================================
    # Check each each peak pixel to see if it belongs to a bowtie
    # using the classifier
    # =============================================================================
    
    clf=joblib.load(clf_dir+'\\XGBC_img_classifier.pkl')
    p_crit=0.8 #yields the highest classifier recall without suffering a serious drop in precision
    
    bowties=[]
    for pixel in maxidx:
        features=dp.extract_features_XGBC(img0,img45,pixel)
        p=clf.predict_proba(features)
        if p>=p_crit:
            bowties.append(1)
        else:
            bowties.append(0)
    
    # =============================================================================
    # Place a white box around each positively identified bowtie
    # =============================================================================
    loccount=0
    for k in maxidx:        
        pf.append(1)
        imgloc.append(count)
        subloc.append(loccount)
        peakpixel.append(k)
        
        #This places a white box around each pixel of interest
        img0=basic.boxpoint(img0,k,val)
        img45=basic.boxpoint(img45,k,val)

        loccount+=1
        continue
    

    
    '''~~~~~~~~~~~~~ANNOTATE BOXES~~~~~~~~~~~~~~~'''
    plt.gray()
    plt.close('all')
    
    fig,ax=plt.subplots()
    ax.imshow(img0)
    
    loccount=0
    for k in maxidx:        
        offsetbox = TextArea(str(loccount), minimumdescent=False)
        ab = AnnotationBbox(offsetbox,(k%xdim,k//xdim), xybox=(0,25), xycoords='data', boxcoords="offset points")
        ax.add_artist(ab)
                   
        loccount+=1
    
    os.chdir(imgsavedir)
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
    
    os.chdir(imgsavefile)
    plt.savefig(str(count)+'_45.png')
    
    #os.chdir(imgsavefile)
    #plt.imsave(str(count)+'_0.png',img0)
    #plt.imsave(str(count)+'_45.png',img45)
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

    
    count+=1
    
    print(wafer+': '+str(count)+' of '+str(len(filenames)))
    
    #if count>105:
    #    break