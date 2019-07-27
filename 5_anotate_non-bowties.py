# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:28:41 2018

@author: Logan Rowe

The purpose of this script is to:
    Generate 5x images from dt1 files
    Post process the images by applyinig a subtraction image and resetting hot pixels
    Subdivide the image into 16 equally sized smaller images
    Select the most intense (based on shear max image) pixel from each sub-image
    Plot the full image with the top 16 pixels (1 from each sub-image) boxed and annotated
    Save the image in the designated folder

Bowties are the result of the wafer's bulk residual stress acting on microcracks.
As such, the center of a bowtie registers a very high shear max stress.
Boxing and annotating high intensity pixels facilitates the process of identifying
potential bowties that can be used to train the machine learning classifier.  
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


xdim,ydim=640,480

psdr=False
regen=True

       
#FOLDERS
savefile='E:\\cSi Wafer Data\\DeltaVision Scripts\\Wafer Images'
subfile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Subtraction_Images'
hotpixfile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Light Levels\\Hot_Pixel_Lists'
lightlevelfile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Light_Levels'
savedir2='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images'

  
#WAFERS TO BE ANALYZED
os.chdir(subfile)
if psdr==True:
    wafers=glob.glob('*cSi*')
else:
    wafers=[]
    for i in glob.glob('*'):
        if 'cSi' not in i:
            wafers.append(i)

wafers=['17','27','50'] #TESTING PURPOSES ONLY CAN DELETE THIS LINE

'''
We are interested in saving images of boxed regions that do not contain bowties
Given the sparse nature of bowties we will box the same pixels in each image
(1 region per sub image) and then go back and select the ones that do not contain a bowtie

The pixel selected (for simplicity) will be the central pixel in each subimage region
'''
maxidx=[] #pixels around wich a box will be placed

xpix=[(i+0.5)*160 for i in range(4)]
ypix=[(i+0.5)*120 for i in range(4)]

for i in xpix:
    for j in ypix:
        maxidx.append(int(i+(j*640)))

maxidx=np.sort(maxidx)

'''
A=np.full((480,640),0)
for i in maxidx:
    A[i//640][i%640]=1
plt.gray()
plt.imshow(A)
'''

for wafer in wafers:
    imgsavefile=savefile+'\\'+wafer+'\\'+'Non-bowties' #This is where the wafer images will be stored
    if os.path.exists(imgsavefile)!=True:
        os.makedirs(imgsavefile)
    
    #SKIP ALREADY IMAGED WAFERS
    if os.path.exists(savefile+'\\'+wafer) and regen==False:
        print("Skipped "+wafer)
        continue
    
    #LOAD SUBIMG
    os.chdir(subfile+'\\'+wafer)
    sub0=np.load('sub0_low.npy')
    sub45=np.load('sub45_low.npy')
    
    #LOAD LIGHT DATA
    os.chdir(lightlevelfile)
    W,AA,AS,SA,SS,MM=np.genfromtxt('WaferLight_AA_AS_SA_SS_MM.txt',delimiter=',',unpack=True).tolist()
    
    try:
        idx=W.index(int(wafer))
        AA,AS,SS,SA=AA[idx],AS[idx],SS[idx],SA[idx]
    except:
        pass

    #LOAD HOT PIXEL LIST
    os.chdir(lightlevelfile+'\\Hot_Pixel_Lists')
    hotpix=np.genfromtxt(wafer+'_hotpix.txt').tolist()
    hotpix=[int(i) for i in hotpix]
    
    if psdr==True:
        datafile='E:\\cSi Wafer Data\\1_'+wafer+'_cSi\\dt1'
    else:
        datafile='E:\\cSi Wafer Data\\1_'+wafer+'\\dt1_'
        datafile='D:\\Talbot Runs\\1_'+wafer+'\\dt1_'

    os.chdir(datafile)
    
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
        if count%8!=0:
            print('skipped',count)
            count+=1
            continue
        
        if count>300:
            rows=zip(imgloc,subloc)
            basic.WriteRows(rows,savedir2,str(wafer)+'_imgloc_subloc manual ID other')
            break
        
        os.chdir(datafile)
        img0,img45,imgL=basic.formimg(i,datafile)
        img0-=sub0
        img45-=sub45
        
        os.chdir(datafile)
        #CHECK TO SEE IF IMAGE PASSES LIGHT LEVEL TEST        
        PF=basic.lightmask(np.mean(imgL),np.std(imgL),AA,SA,AS,SS)
        
        #IF IMAGE FAILS THE LIGHT TEST THEN SKIP THAT IMAGE (CONTINUE)
        if PF==False:
            pf.append(0)
            imgloc.append(count)
            subloc.append(0)
            peakpixel.append(0)
                        
            count+=1
            continue
        
        #RESET ALL HOT PIXELS TO IMAGE AVERAGE
        val0=np.mean(img0)
        val45=np.mean(img45)
        for j in hotpix:
            img0[j//xdim][j%xdim]=val0 
            img45[j//xdim][j%xdim]=val45
            #image[y][x]=value where y and x are calculated from image width in pixels (xdim) and pixel index (j)
               
        #NOW MAKE SHEAR MAX IMAGE
        #imgM=(img0**2+img45**2)**0.5  
        imgM=basic.shear_max_img(img0,img45)
        
        
        #IF IMAGE DOES PASS SUBDIVIDE IMAGE AND GET LOCATION OF MAX SHEAR MAX
        #dx,dy=160,120 #To make 16 images per 5x shear max image
        #subimages=basic.subsub(imgM,xdim,ydim,dx,dy)
    
        #RECORD LOCATIONS OF PEAK RETARDATIONS
        #localmaxes=[np.argmax(k) for k in subimages]
        #maxidx=[basic.LocalToGlobalIdx(k,m,dx,dy,xdim//dx) for (k,m) in zip(localmaxes,range(0,len(localmaxes)))]
        
        val=np.max(imgM)
        
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
        
        plt.gray()
        plt.close('all')
        
        '''~~~~~~~~~~~~~ANNOTATE BOXES~~~~~~~~~~~~~~~'''
        fig,ax=plt.subplots()
        ax.imshow(img0)
        
        loccount=0
        for k in maxidx:        
            offsetbox = TextArea(str(loccount), minimumdescent=False)
            ab = AnnotationBbox(offsetbox,(k%xdim,k//xdim), xybox=(0,25), xycoords='data', boxcoords="offset points")
            ax.add_artist(ab)
                       
            loccount+=1
        
        os.chdir(imgsavefile)
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
        
        rows=zip(imgloc,subloc)
        basic.WriteRows(rows,savedir2,str(wafer)+'_imgloc_subloc manual ID other')
        
        #if count>105:
        #    break
    