# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:08:36 2018

@author: Logan Rowe

This script generates a shear 0 and a shear 45 subtraction image for each wafer.
The subtraction image is the average of N=108 shear 0 and shear 45 images.

The images selected to make the subtraction images are those that have the lowest
shear max retardation.  The bulk shear max retardation is present due to both the 
wafer and polychromatic nature of the light source.  By selecting the images with
the lowest shear max value we aim to create a subtraction image that will remove
optical aberrations while having a minimal effect on the retardation induced by the wafer.
"""

import os
import numpy as np
from myscripts3 import basic_tools as basic
import glob
import matplotlib.pyplot as plt

savefile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Subtraction Images'
waferfile='D:\\Talbot Runs'
lightlevelfile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Light Levels'

#REGENERATE ALREADY EXISTING SUBTRACTION IMAGES?
regen=False

#AS CUT OR PSDR (ETCHED)
psdr=False

xdim,ydim=640,480
        
os.chdir(waferfile)
waferlist=[]

#UNCOMMENT THIS SECTION TO RUN ALL WAFERS AT ONCE (AND DELETE waferlist below)
''' 
for i in glob.glob('1_*'):
    if psdr==True:
        if 'cSi' in i:
            waferlist.append(i.split('_')[1])
    else:
        if 'cSi' not in i:
            waferlist.append(i.split('_')[1])
'''

#OR RUN ONLY SELECTED WAFERS
waferlist=['50']  

#BUILD DICTIONARY OF LOWEST SHEAR MAX IMAGES THAT PASS LIGHT TEST FOR EACH WAFER
#USE THESE IMAGES FOR THE SUBTRACTION IMAGE SINCE THE LOWEST SHEAR MAX SHOULD CONSIST
#OF ONLY OPTICAL ABERRATIONS (OR AS CLOSE TO AS WE WILL GET)
imagedict={}
os.chdir(lightlevelfile+'\\Data')
files=glob.glob('*d45.t*')
for i in files:
    pflist=[]
    mMlist=[]
    locationlist=[]
    
    
    #Unpack (wafer)_aveL_stdL_dM_d0_d45.txt into average light intensity (for each image)
    #standard deviation of pixel intensities (for each image), mean shear max value (for each image)
    #mean shear 0 value (for each image) and mean shear 45 value (for each image)
    AL,SL,mM,m0,m45=np.genfromtxt(i,delimiter=',',unpack=True).tolist()
    AA=np.mean(AL)
    AS=np.mean(SL)
    SA=np.std(AL)
    SS=np.std(SL)
    
    loc=0
    for (j,k,l) in zip(AL,SL,mM):
        if basic.lightmask(j,k,AA,SA,AS,SS)==True: #Test whether image quality is acceptable (1) or poor (0)
            pflist.append(1)
        else:
            pflist.append(0)
    
        locationlist.append(loc)
        
        mMlist.append(l)
        
        loc+=1
    
    imagedict[i.split('_')[0]]=zip(locationlist,pflist,mMlist)

'''
#Test what fraction of images are not rejected because of light (each out of 3300)
for i in imagedict:
    count=0
    for j in imagedict[i]:
        count+=j[1]
    print(i)
    print(str(count/len(pflist)),'\n')
'''

"""+++"""
#GENERATE SUBTRACITON IMAGE
"""+++"""

count=0
while count<len(waferlist):
    wafer=waferlist[count]
    
    #BUILD LIST OF TOP N IMAGES WHICH PASS LIGHT TEST AND HAVE THE LOWEST MEAN SHEAR MAX
    #loc,pf,mM=map(list,zip(*imagedict[wafer]))     #Unpack dictionary into lists for python 2.7
    loc,pf,mM=list(map(list, zip(*imagedict[wafer]))) #for python 3.7
    sorteddata=sorted(zip(mM,loc,pf))                  #Sorted zip with mM first
    
    #BUILD IMAGE LIST OF LOWEST N IMAGES
    N=108
    images=[]
    mM_=[]
    
    count_=0
    appendcount=0
    for (i,j,k) in sorteddata:
        if k==1:
            images.append(j)
            mM_.append(i)
            appendcount+=1
        if appendcount==N:
            break
        count_+=1 #No matter whether image passed or not
                
    
    savedir=savefile+'\\'+wafer
    
    print("Starting Wafer",wafer,':',str(count+1),'of',str(len(waferlist)))
    
    if regen==False: #Skips subtraction image generation if file has already been generated for that wafer
        if os.path.exists(savedir+'\\sub0_low.npy')==True:
            count+=1
            continue
        
    if psdr==True:
        datafile=waferfile+'\\1_'+wafer+'_cSi\\dt1'
    else:
        datafile=waferfile+'\\1_'+wafer+'\\dt1_'
        
    
    #LOAD LIST OF HOT PIXELS FOR WAFER
    os.chdir(lightlevelfile+'\\Hot Pixel Lists')
    hotpix=np.genfromtxt(wafer+'_hotpix.txt').tolist()
    hotpix=[int(i) for i in hotpix]
    
    os.chdir(datafile)
    
    filenames=[]
    for i in glob.glob('*.dt1'):
        filenames.append(i)
        
    img0bar,img45bar,imgL=basic.formimg(filenames[int(images[0])],datafile)

    k=0
    while k < len(images):
        img0_,img45_,imgL_=basic.formimg(filenames[int(images[k])],datafile)
        
        val0=np.mean(img0_)
        val45=np.mean(img45_)
        
        #RESET HOT PIXELS TO MEAN OF IMAGE
        img0_=basic.resetpixels(img0_,hotpix,val0)
        img45_=basic.resetpixels(img45_,hotpix,val45)
        
        img0bar+=img0_
        img45bar+=img45_
        if k%36==0:
            print(str(k)+' of '+str(len(images)))
        k+=1
        
    
    subimg0=img0bar/(len(images))
    subimg45=img45bar/(len(images))
    
    if os.path.exists(savedir)!=True:
        os.makedirs(savedir)
    
    
    os.chdir(savedir)
    np.save('sub0_low.npy',subimg0)
    np.save('sub45_low.npy',subimg45)
    
    count+=1



plt.close('all')
plt.gray()

fig,(ax1,ax2)=plt.subplots(1,2,sharey='row')
ax1.imshow(subimg0)
ax2.imshow(subimg45)
ax1.set_title(r'$\delta_0$$^{(sub)}$')
ax2.set_title(r'$\delta_{45}$$^{(sub)}$')
