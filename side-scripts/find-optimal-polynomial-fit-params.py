# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:21:02 2019

@author: Logan Rowe

Check whether a rasonable approximation of the shear 0 and shear 45
circle sweeps can be made via a polynomial fit

If so then polynomial coefficients can be used as a metric for the machine
learning training data

Results: Negative
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,savgol_coeffs
import glob

example_bowtie_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\side-scripts\\example-bowties'
bowties=glob.glob(example_bowtie_dir+'\\*shear45*')
bowties=['_'.join(file.split('\\')[-1].split('_')[1:5]) for file in bowties]

data=np.load('..\\Wafer_Images\\bowtie-training-data\\wafer_loc_subloc_pixel_thetaM_shear0_shear45_bow-bool.npy')
savedir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\side-scripts\\polynomial-fits'

#Wafer, Loc, Subloc, Pixel, ThetaM, Shear 0 (72), Shear 45 (72), bow boolean
print('_'.join([str(int(i)) for i in data[0][:4]]))

#Create dictionary to quickly locate index of the bowtie of interest
data_dict={}
index=0
for row in data:
    data_dict['_'.join([str(int(i)) for i in row[:4]])]=index
    index+=1

#Remove any images that were rejected before making the data set
for i in bowties:
    try:
        print(data_dict[i])
    except KeyError:
        del bowties[bowties.index(i)]
        pass
    
for i in bowties:
    W,L,S,P=i.split('_')
    theta=float(data[data_dict[i]][4])
    thetas=[float(theta+i) for i in np.linspace(0,2*np.pi,72)]
    shear0=[float(val) for val in data[data_dict[i]][5:77]]
    shear45=[float(val) for val in data[data_dict[i]][77:-1]]
    
    power=5
    
    shear0_params=np.polyfit(shear0,thetas,power)
    shear45_params=np.polyfit(shear45,thetas,power)
    
    shear0_=np.poly1d(shear0_params)
    shear45_=np.poly1d(shear45_params)
    
    shear0_=shear0_(thetas)
    shear45_=shear45_(thetas)
    
    
    plt.figure()
    plt.plot(thetas,shear0,'r--',linewidth=2)
    plt.plot(thetas,shear45,'b--',linewidth=2)
    plt.plot(thetas,shear45_,'k-')
    plt.plot(thetas,shear0_,'k-')
    plt.xlabel(r'$\theta$ [rad]')
    plt.ylabel('Intensity [arb.]')
    plt.title('Wafer '+W+' Loc '+L+' Sub '+S+' Pixel '+P+' Bowtie '+str(bool(1))+'\nPower '+str(power))
    plt.legend(['Shear 0 Circle Sweep','Shear 45 Circle Sweep','Sav-Gol Fit'])
    name=str(power)+'_'+i+'.png'
    
    if os.path.exists(savedir+'\\'+str(power))!=True:
        os.makedirs(savedir+'\\'+str(power))
    
    break
    plt.savefig(savedir+'\\'+str(power)+'\\'+name)
    
    plt.close('all')