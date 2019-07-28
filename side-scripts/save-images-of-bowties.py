# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:32:30 2019

@author: Logan Rowe
"""


import numpy as np
import os
import sys

os.chdir('..')

from myscripts3 import basic_tools as basic


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

datafile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowtie-training-data'
bowtiefile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowties'
nonbowtiefile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\non-bowties'

savefile='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\side-scripts\\bowtie-plots'

X=np.load(datafile+'\\wafer_loc_subloc_pixel_shear0_shear45_bow-bool.npy')

print(X.shape)

regen=False #to not save plots for plots that already exist


count=0
for bow in X:
    print(str(count)+'/'+str(len(X)))
    count+=1
    
    W,L,S,P=[int(val) for val in bow[:4]]
    T=float(bow[4])
    shear0=[float(val) for val in bow[5:77]]
    shear45=[float(val) for val in bow[77:-1]]
    bowtie=int(bow[-1])
    
    name='_'.join([str(bowtie),str(W),str(L),str(S),str(P)])
    
    if os.path.exists(savefile+'\\'+name+'_parametric.png')==True:
        print('Image already exists')
        count+=1
        continue
    
    #PARAMETRIC PLOT OF SHEAR 0 AND SHEAR 45 CIRCLE SWEEPS
    plt.figure(1)
    plt.plot(shear0,shear45,'r-')
    plt.xlabel('Shear 0')
    plt.ylabel('Shear 45')
    plt.title('Wafer '+str(W)+' Loc '+str(L)+' Sub '+str(S)+' Pixel '+str(P)+' Bowtie '+str(bool(bowtie)))
    plt.savefig(savefile+'\\'+name+'_parametric.png')
    plt.close('all')
    
    #PLOT OF SHEAR 0, SHEAR 45 AND SHEAR MAX CIRCLE SWEEPS
    plt.figure(2)
    shearM=basic.shear_max_img(np.array(shear0),np.array(shear45))
    thetas=[i*180/np.pi for i in np.linspace(T,(2*np.pi)+T,72)]
    plt.plot(thetas,shear0,'g-')
    plt.plot(thetas,shear45,'b-')
    plt.plot(thetas,shearM,'r-')
    plt.xlabel('Angle of Linescan [deg]')
    plt.ylabel('Retardation [arb.]')
    plt.title('Wafer '+str(W)+' Loc '+str(L)+' Sub '+str(S)+' Pixel '+str(P)+' Bowtie '+str(bool(bowtie)))
    plt.savefig(savefile+'\\'+name+'_scan.png')
    plt.close('all')

    if bowtie==1:
        os.chdir(bowtiefile+'\\'+str(W))
    else:
        os.chdir(nonbowtiefile+'\\'+str(W))
        
    bow0,bow45=np.load(str(W)+'_'+str(L)+'_'+str(S)+'_'+str(P)+'_0.npy'),np.load(str(W)+'_'+str(L)+'_'+str(S)+'_'+str(P)+'_45.npy')

    plt.gray()
    plt.figure(3)
    plt.imshow(bow0)
    plt.savefig(savefile+'\\'+name+'_shear0.png')
    plt.close('all')
    
    plt.figure(4)
    plt.imshow(bow45)
    plt.savefig(savefile+'\\'+name+'_shear45.png')
    plt.close('all')