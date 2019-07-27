# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 00:02:58 2019

@author: Logan Rowe
"""

import numpy as np
import os
import sys
import glob
from myscripts3 import basic_tools as basic
import matplotlib.pyplot as plt


bowtie_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowties'
nonbowtie_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\non-bowties'

os.chdir(bowtie_dir)
wafers=glob.glob('*')
print(wafers)

for identity in [('bowtie',bowtie_dir),('nonbowtie',nonbowtie_dir)]:
    for wafer in wafers:
        os.chdir(identity[1]+'\\'+wafer)
        count=0
        for (bow0_file,bow45_file) in zip(glob.glob('*0.npy'),glob.glob('*45.npy')):
            if bow45_file.split('_')[-2]=='0':
                #For unknown reason sometimes {wafer}_0_0_0_45.npy somehow makes 
                #its way into the bowtie dataset even though it is not in the data file
                #this will skip the phantom bowtie
                continue
            bow0,bow45=np.load(bow0_file),np.load(bow45_file)
            plt.imshow(bow45)
            break
            