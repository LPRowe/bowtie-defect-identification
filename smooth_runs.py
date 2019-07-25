# -*- coding: utf-8 -*-
"""
Created on Sat Sep 01 16:29:54 2018

@author: Logan Rowe

Early on in the data acquisition process there were issues with files sporatically
being corrupted or the automation process freezing.  

Wafers folders containing precicely 3234 images were smooth runs and likely collected
after all of the bugs were worked out.  As such I recommend you start your wafer
analysis project using wafers from smooth imaging runs.  

This script will look at all of the wafer folders and save a list of the wafers
that had a clean run.  
"""

import numpy as np
import os
import glob

datafile='D:\\Talbot Runs'

wafers=[]

for i in range(0,96):
    try:
        os.chdir(datafile+'\\1_'+str(i)+'\\dt1_')
        if len(os.listdir(datafile+'\\1_'+str(i)+'\\dt1_'))==3234:
            wafers.append(i)       
        print(i)
    except:
        pass
    
print(len(wafers))

file='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\promising_wafers.txt'
fd=open(file,'w')
fd.close()
    
with open(file,'a') as fd:
    for i in wafers:
        fd.write(str(i)+'\n')

    