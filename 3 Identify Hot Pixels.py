# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:56:33 2018

@author: Logan Rowe
# -*- coding: utf-8 -*-

Identify hot pixels from top pix list

Top pix contains a list of the location of the top 20 pixels (by intensity) 
for each 5x (microscopic) IR-GFP image.  This script will look at the top pixel
in each image and if a pixel occurs as the top pixel more than 4 times it is marked
as a hypersensitive (hot) pixel.  

All pixels denoted as hypersensitive are then removed from the list of the top 20 pixels
and the process is repeated until all hypersensitive pixels have been identified.

Note: It is rare for a hypersensitive pixel to only occur 5-10 times.  If a pixel
is hypersensitive it will occur many times, sometimes hundreds of times over the
course of 3234 images.  

Note: Hypersensitive pixels occur more frequently with long imaging sessions.  
Turning the camera off and allowing it to cool to ambient temperature greatly
mitigates the issue of hot pixels.  
"""

import numpy as np
import os
import glob

savefile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Light_Levels\\Hot_Pixel_Lists'
datafile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Light_Levels\\Data'
waferdatafile='E:\\cSi Wafer Data\\DeltaVision Scripts\\Wafer Data Files'
psdr=False #etched=True and as-cut wafers=False


#BUILD WAFER LIST
waferlist=[]
os.chdir(waferdatafile)

if psdr==True:
    for i in glob.glob('*cSi*'):
        waferlist.append(i.split('_')[1])
else:
    for i in glob.glob('*'):
        if 'cSi' not in i:
            waferlist.append(i.split('_')[1])

#OR SPECIFY WHICH WAFER(S) YOU WANT TO RUN WITH THIS SCRIPT        
waferlist=['50']
            

for wafer in waferlist:
    print("Starting "+str(wafer))
    os.chdir(datafile)
    
    data=wafer+'_TopPix.txt'
    
    hotpixels=[]
    
    data=np.genfromtxt(data,delimiter=',')
    
    dataset=list(set(data[:,0]))
    
    for i in dataset:
        if data[:,0].tolist().count(i)>=4:
            hotpixels.append(i)
        
    #CONVERT TO LIST TO OBSERVE FIRST PIXEL IN LIST AND REMOVE IF IT IS HOT
    datalist=data.tolist()
    while True:
        count=0 #Track removed hot pixels
        for i in datalist:
            try:
                while i[0] in hotpixels:
                    del i[0]
                    count+=1
            except:
                break
        
        print(count)
                
        #REBUILD ARRAY AND CHECK FOR NEW HOT PIXELS
        pixlist=[i[0] for i in datalist if len(i)!=0]
        for i in pixlist:
            if pixlist.count(i)>=4:
                hotpixels.append(i)
          
        if count==0:
            break
    
    hotpixels=list(set(hotpixels))
    hotpixels=[int(i) for i in hotpixels]
    
    print(hotpixels)
    print(len(hotpixels))
    
    #SAVE HOT PIXEL LIST
    os.chdir(savefile)
    name=wafer+"_hotpix.txt"
    fd=open(name,'w')
    for i in hotpixels:
        fd.write(str(i))
        fd.write('\n')
    fd.close()

