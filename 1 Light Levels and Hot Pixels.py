# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:23:46 2018

@author: Logan Rowe

This script will collect the following information from each wafer:

1) Average of Average Light Value
2) Average of STD Light Value
3) STD of Average Light Value
4) STD of STD Light Value

5) List of top 20 highest value pixels for each shear max image
6) Build hot pixel list from (5)
    
The purpose of collecting this information is to identify images that are of
the wafer mask and to detect hypersensitive (hot) pixels
"""

import numpy as np
import os
import glob
from myscripts3 import analysis_tools as at

#INPUT INFORMATION HERE
psdr=False #Is the wafer etched (psdr) or as cut? PSDR is true if wafer name starts with 00## or 02## where as as cut (psdr=False) are just ##
trunc=3300 #how many images to stop at (cut off top dark space) typically 3300 for etched and 3234 for as-cut
datafile='E:\\cSi Wafer Data\\DeltaVision Scripts\\Wafer Data Files'
savefile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Light_Levels'

regen=False #IF TRUE WILL ANALYZE WAFERS THAT HAVE ALREADY BEEN ANALYZED BEFORE IF FALSE IT WILL SKIP THESE WAFERS


#DATA FILE AND SAVE FILE (BELOW) CAN BE DELETED LATER THIS IS TEMPORARY TO TEST THE SCRIPTS
datafile='D:\\Talbot Runs'


xdim,ydim=640,480


#UNCOMMENT THIS SECTION FOR MULTIWAFER ANALYSIS
'''
waferlist=[]
os.chdir(datafile)
if psdr==True:
    for i in glob.glob('*cSi*'):
        waferlist.append(i.split('_')[1])
else:
    for i in glob.glob('*'):
        if 'cSi' not in i:
            waferlist.append(i.split('_')[1])
'''
        
waferlist=['50'] #IF ANALYZING ONLY SPECIFIC WAFERS
    

count=0
while count<len(waferlist):
    wafer=waferlist[count]

    if os.path.exists(savefile+'\\'+wafer+'_aveL_stdL_dM_d0_d45.txt') and regen==False:
        count+=1
        print("Skipped",str(wafer))
        continue    
    
    if psdr==True:
        dt1file='E:\\cSi Wafer Data\\DeltaVision Scripts\\Wafer Data Files\\1_'+wafer+'_cSi\\dt1'
    else:
        dt1file='E:\\cSi Wafer Data\\DeltaVision Scripts\\Wafer Data Files\\1_'+wafer+'\\dt1_'
        dt1file='D:\\Talbot Runs\\1_'+wafer+'\\dt1_'
        
    os.chdir(dt1file)
    
    filenames=[]
    for i in glob.glob('*.dt1'):
        filenames.append(i)
        
    filenames=filenames[:trunc] #Truncate to top of wafer
    
    #DATA TO BE GATHERED (See items 1-4 at top)
    avelight=[]
    stdlight=[]
    dM=[]
    d0=[]
    d45=[]
    
    pix=[]   
    
    imgcount=0
    for i in filenames:
        os.chdir(dt1file)
        at.formimg(i)
        img0=at.img0
        img45=at.img45
        imgL=at.imgL
        imgM=(img0**2+img45**2)**0.5 #Not exactly correct but okay for hunting hot pixels and getting average value
        
        #LIGHT DATA
        avelight.append(np.mean(imgL))
        stdlight.append(np.std(imgL))
        
        dM.append(np.mean(imgM))
        d0.append(np.mean(img0))
        d45.append(np.mean(img45))
        
        #TOP SHEAR MAX PIXELS
        idx=[]
        while len(idx)<20:
            idx.append(np.argmax(imgM))
            imgM[idx[-1]//xdim,idx[-1]%xdim]=0
        
        if imgcount%50==0:
            print(wafer+': '+str(imgcount)+' of '+str(len(filenames)))
            
        pix.append(idx)
        
        imgcount+=1
    
    os.chdir(savefile)
    
    #SEPARATE HOT PIX AND LIGHT VALUES FOR EASY DATA ANALYSIS LATER
    name=wafer+'_aveL_stdL_dM_d0_d45.txt'
    fd=open(name,'w')
    for (i,j,k,l,m) in zip(avelight,stdlight,dM,d0,d45):
        fd.write(str(i)+','+str(j)+','+str(k)+','+str(l)+','+str(m))
        fd.write('\n')
    fd.close()
    
    name=wafer+'_TopPix.txt'
    fd=open(name,'w')
    for i in pix:
        fd.write(str(i)[1:-1])
        fd.write('\n')
    fd.close()  
    
    count+=1