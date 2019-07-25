# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:52:01 2018

@author: Logan Rowe

Saves a comma delimited list of the wafer number, (1), (2), (3), (4)
for each wafer in the designated savefile

1) Average of Average Light Value
2) Average of STD Light Value
3) STD of Average Light Value
4) STD of STD Light Value
5) Average of Shear Max Before Subtraction Image

These values will be used to help filter out poor quality images
"""

import numpy as np
import os
import glob


datafile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Light_Levels\\Data'
savefile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Light_Levels'
os.chdir(datafile)



filelist=[]
for i in glob.glob('*d45.t*'):
    filelist.append(i)

waferlist=[i.split('_')[0] for i in filelist]
AA=[] #SEE ITEMS (1-4) AT TOP
AS=[]
SA=[]
SS=[]
MM=[]


for i in filelist:
    aveL,stdL,dM,d0,d45=np.genfromtxt(i,delimiter=',',unpack=True)
    AA.append(np.mean(aveL))
    AS.append(np.mean(stdL))
    SA.append(np.std(aveL))
    SS.append(np.std(stdL))
    MM.append(np.mean(dM))
    
#WRITE VALUES
os.chdir(savefile)
name='WaferLight_AA_AS_SA_SS_MM.txt'
fd=open(name,'w')
rows=zip(waferlist,AA,AS,SA,SS,MM)
for row in rows:
    count=0
    for i in row:
        if count<(len(row)-1):
            fd.write(str(i)+',')
        else:
            fd.write(str(i))
        count+=1
    fd.write('\n')
fd.close()
