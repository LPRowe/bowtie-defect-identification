# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:12:50 2017

@author: Logan Rowe
"""

from __future__ import division
from array import *
import numpy as np
import csv
import sys
import os
import scipy.special as sp
import pickle
from myscripts3 import linescans as scan

xdim=640
ydim=480

BadIndices=[]

def diradj(d0,d45,theta_diradj):
    '''
    Returns correct d0 and d45 based on direction adjustment
    This is equivalent to altering the direction adjustment setting in the IR-GFP
    See IR-GFP manual for more details
    '''
    theta=theta_diradj
    C_0=C(0)
    C_45=C(np.pi/4)
    d45_adj=d45*np.cos(2*theta)+(C_0/C_45)*d0*np.sin(2*theta)
    d0_adj=-(C_45/C_0)*d45*np.sin(2*theta)+d0*np.cos(2*theta)
    return (d0_adj,d45_adj)

def sigratio(KR,beta,alpha):
    '''
    This is the ratio of principal stresses based on KI/KII 
    it is derived from Shlyannikov's paper on Mode Mixity Parameters
    '''
    sr=(1/np.tan(beta-alpha))*(KR*np.sin(beta-alpha)-np.cos(beta-alpha))/(KR*np.cos(beta-alpha)+np.sin(beta-alpha))
    return sr

def ConvList(From,To):
    for i in From:
        To.append(i)

#STRESS OPTIC COEFFICIENT FOR SILICON
def C(theta):
    '''
    Stress Optic Coefficient for Silicon
    theta=0 along the 100 direction of silicon
    Piezo optic coefficients obtained from Danyluk et al. 2014
    '''
    p44=6.5e-13 #Pa-1
    p11_12=9.88e-13 #Pa-1
    n_0=3.54 #Ordinary refractive index
    C=(n_0**3/2)*(((np.sin(2*theta)/p44)**2+(np.cos(2*theta)/p11_12)**2)**(-0.5))
    return C
    
def d0_calc(KI,KII,theta,alpha):
    '''
    KI in pa*m^0.5 theta,alpha radians and returns retardation in meters
    Expected shear 0 retardation from a microcrack with given stress intensity factors (KI,KII) that has a crack plane along angle alpha from the 100 driection
    *Assumes the silicon wafer is 170 um thick and measurement is taken 5 um radius from the crack tip
    '''
    d=(5.841573026*10**(-13)*KI*np.sin(alpha)*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.sin((3/2)*theta)*np.cos(alpha)-5.841573026*10**(-13)*KI*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.cos((3/2)*theta)*np.cos(alpha)**2+5.841573026*10**(-13)*KII*np.sin(alpha)*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.cos((3/2)*theta)*np.cos(alpha)+5.841573026*10**(-13)*KII*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.sin((3/2)*theta)*np.cos(alpha)**2+2.920786513*10**(-13)*KI*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.cos((3/2)*theta)+5.841573026*10**(-13)*KII*np.sin(alpha)*np.sin((1/2)*theta)*np.cos(alpha)-2.920786513*10**(-13)*KII*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.sin((3/2)*theta)-5.841573026*10**(-13)*KII*np.cos((1/2)*theta)*np.cos(alpha)**2+2.920786513*10**(-13)*KII*np.cos((1/2)*theta))
    return(d)
    
def d45_calc(KI,KII,theta,alpha):
    d=-8.879190998*10**(-13)*KI*np.sin(alpha)*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.cos((3/2)*theta)*np.cos(alpha)-8.879190998*10**(-13)*KI*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.sin((3/2)*theta)*np.cos(alpha)**2+8.879190998*10**(-13)*KII*np.sin(alpha)*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.sin((3/2)*theta)*np.cos(alpha)-8.879190998*10**(-13)*KII*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.cos((3/2)*theta)*np.cos(alpha)**2+4.439595499*10**(-13)*KI*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.sin((3/2)*theta)-8.879190998*10**(-13)*KII*np.sin(alpha)*np.cos((1/2)*theta)*np.cos(alpha)+4.439595499*10**(-13)*KII*np.cos((1/2)*theta)*np.sin((1/2)*theta)*np.cos((3/2)*theta)-8.879190998*10**(-13)*KII*np.sin((1/2)*theta)*np.cos(alpha)**2+4.439595499*10**(-13)*KII*np.sin((1/2)*theta)
    return(d)

def dM_calc_normalized(KI,KII,theta):#Normalized dM
    KR=KI/KII
    dM=((1-((3-KR**2)*(0.25*np.sin(theta)**2))+(0.5*KR*np.sin(2*theta))))**0.5
    return dM

    
def beta_true(d0,d45):
    '''
    Calculates the direction of the first principal stress accounting for C0,C45
    Note: dirimg_calc and dirpixel_calc will match deltavision perfectly, but because of anisotropy in stress
    optic coefficient this is the true direction of first principal stress
    '''
    C_45=C(np.pi/4)
    C_0=C(0)
    A=(d45*C_45)/(d0*C_0)
    beta=np.arctan(A-((1+A**2)**0.5))
    return beta

def dirimg_true(img0,img45):
    C_45=C(np.pi/4)
    C_0=C(0)
    A=(img45*C_45)/(img0*C_0)
    dirimg=np.arctan(A-((1+A**2)**0.5))
    return dirimg
    
def dM_true(d0,d45):
    beta=beta_true(d0,d45)
    C_0=C(0)
    C_B=C(beta)
    C_45=C(np.pi/4)
    dM=C_B*((d45/C_0)**2+(d0/C_45)**2)**0.5
    return dM
    
    
def imgM_true(img0,img45,m0,m45):
    '''
    Calculates the shear max image while accounting for C0, C45 and C(Beta)
    beta is calculated from mean value here, it should be done pixel by pixel but this 
    function will be normalized so for the sake of speed beta(m0,m45)
    '''
    beta=beta_true(m0,m45)
    C_0=C(0)
    C_B=C(beta)
    C_45=C(np.pi/4)
    imgM=C_B*((img45/C_0)**2+(img0/C_45)**2)**0.5
    return imgM
    
def read_circle_sweep_txt(file):
    filename=file
    fd=open(filename,'r')
    lines=fd.readlines()
    KI=[float(i.split('"')[0].split(',')[0]) for i in lines]
    KII=[float(i.split('"')[0].split(',')[1]) for i in lines]
    A=[float(i.split('"')[0].split(',')[2]) for i in lines]
    
    dM=[]
    
    for i in lines:
        dM_=i.split('"')[1].split('[')[1].split(']')[0]
        dM_=dM_.split(',')
        a=[]
        for k in dM_:
            a.append(float(k))
        dM.append(a)
        
    return (KI,KII,A,dM)

def read_circle_sweep_txt_true(file):
    filename=file
    fd=open(filename,'r')
    lines=fd.readlines()
    KI=[float(i.split('"')[0].split(',')[0]) for i in lines]
    KII=[float(i.split('"')[0].split(',')[1]) for i in lines]
    A=[float(i.split('"')[0].split(',')[2]) for i in lines]
    
    dM=[]
    d0=[]
    d45=[]
    
    for i in lines:
        dM_=i.split('"')[1].split('[')[1].split(']')[0]
        dM_=dM_.split(',')
        a=[]
        for k in dM_:
            a.append(float(k))
        dM.append(a)
        
        d0_=i.split('"')[3].split('[')[1].split(']')[0]
        d0_=d0_.split(',')
        a=[]
        for k in d0_:
            a.append(float(k))
        d0.append(a)
        
        d45_=i.split('"')[5].split('[')[1].split(']')[0]
        d45_=d45_.split(',')
        a=[]
        for k in d45_:
            a.append(float(k))
        d45.append(a)
        
    return (KI,KII,A,dM,d0,d45)
    
    
def dirimg_calc(img0,img45): 
    '''
    Calculation Validated by comparison between Python and DeltaVision Software
    This calculation of the first principal stress is derived from the eigenvectors of the stress tensor
    Then shear0 and shear45 were subbed into sh0=-s12 and sh45=0.5*(s11-s22) to achieve the following
    SEE MAPLE: RA/2018/PRINC STRSS DIR/1 for  more details
    '''
    beta=-np.arctan((-img45+(img0**2+img45**2)**0.5)/(img0))
    return beta

def dirpixel_calc(d0,d45):
    '''
    This calculation of the first principal stress is derived from the eigenvectors of the stress tensor
    Then shear0 and shear45 were subbed into sh0=-s12 and sh45=0.5*(s11-s22) to achieve the following
    SEE MAPLE: RA/2018/PRINC STRSS DIR/1 for  more details
    '''
    beta=-np.arctan((-d45+np.sqrt(d0**2+d45**2))/(d0))
    return beta

def formimg(filename):
    global img0,img45,imgL
    """
    Reads a .dt1 file and returns imgL, img0 and img45 as numpy arrays.
    
    All images are 640 by 480 arrays
    
    imgL: infrared transmission (Light Image)
    img0: shear 0 image 
    img45: shear 45 image
    
    See IR-GFP manual or Horn et al. 2005 for more details about shear images
    """
    
    filestring = open(filename,"rb").read()
    idx = filestring.rfind(b'@**@Data')
    splitstring=filestring[idx+34:]
    f=array('f')
    f.fromstring(splitstring[:])

    
    f=np.asarray(f)

    #Separate into 3 image types (L,45,0)
    imgLight = f[:xdim*ydim].reshape(ydim,xdim)
    
    imgL=imgLight


    img45 = f[xdim*ydim:2*xdim*ydim].reshape(ydim,xdim)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        img45_ = np.true_divide(img45,imgLight)
        img45_[img45_ == np.inf] = 1
        img45_ = np.nan_to_num(img45_)  
        
    img45=img45_
    
    img0 = f[2*xdim*ydim:3*xdim*ydim].reshape(ydim,xdim)        
    
    #Create dt1 images for 0 and 45 by 45/L and 0/L
    with np.errstate(divide='ignore', invalid='ignore'):
        img0_ = np.true_divide(img0,imgLight)
        img0_[img0_ == np.inf] = 1
        img0_ = np.nan_to_num(img0_)  
    
    img0=img0_
    
    return img0,img45,imgL

       
            
def WriteRows(rows,filepath,name,tab=False):
    '''
    Takes in zip(list1,list2,list3,list4,...) writes as a comma delimited .txt file
    
    rows: zipped lists
    filepath: string - where to save the text file
    name: string - what to name the text file (omit .txt)
    tab=False defaults to comma delimiter
    tab=True tab delimiter
    '''
    csvfile = filepath + '\\' + name + '.txt'
    with open(csvfile, "w") as output:
        if tab!=True:
            writer = csv.writer(output,lineterminator='\n')
        else:
            writer = csv.writer(output,delimiter='\t',lineterminator='\n')
        for row in rows:
           writer.writerow(row)
           
def rebin(a,shape):
    '''
    digitally alter the resolution of array a by changing its shape
    
    a.shape
    >>>(480,640)
    
    reduced_resolution_image=rebin(a,(240,320))
    '''
    sh=shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)
    
def LocalToGlobalIdx(LocalIndex,SubSubImgIndex,Xpixelspersubsub,Ypixelspersubsub,Xsubsubimagesacross):
    '''
    LocalIndex: PIXELS INDEX IN THE SUB-DIVIDED IMAGE (1)
    SubSubImgIndex: THE INDEX OF THE SUB-DIVIDED IMAGE IT IS LOCATED IN (2) 
    Xpixelspersubsub: How many pixels wide is each sub-divided image (3)
    Ypixelspersubsub: How many pixels tall is each sub-divided image (4)
    Xsubimagesacross: How many sub-divided images wide is the image (5) 
    
    (i.e. IF THE SUB IMAGE IS 160 PIXELS ACROSS AND THE FULL IMAGE IS 640 PIXELS ACROSS THEN Xsubsubimagesacross=4)

    
    WILL RETURN THE PIXELS INDEX IN THE FULL IMAGE
    '''
    globalidx_=(SubSubImgIndex//Xsubsubimagesacross)*Xpixelspersubsub*Ypixelspersubsub*Xsubsubimagesacross+(LocalIndex//Xpixelspersubsub)*(Xsubsubimagesacross*Xpixelspersubsub)+(SubSubImgIndex%Xsubsubimagesacross)*Xpixelspersubsub+LocalIndex%Xpixelspersubsub
    return globalidx_

def cu_to_retard(CU,lambda_,thetapk):
    '''
    Takes in Camera Units, lambda [nm], thetapk [CU] and converts to retardation
    '''
    retard=(lambda_*CU/float(2*thetapk*np.pi))
    return retard

def cu_to_Pa(CU,lambda_,thetapk,C,t): #for bulk through thickness measurements
    sigma=lambda_*CU/(2*np.pi*C*t*thetapk)
    return sigma


def read_training_set(location,file):
    '''
    #Takes in training data set file and returns theta 0,theta 45,theta M (orientations of bowties) and average values from linescans for each bow tie starting at t0,t45,tM
    #USE PICKLE TO SAVE AND READ TRAINING SET: TAKES MORE MEMORY BUT IS SIMPLER
    '''
    os.chdir(location)
    fd=open(file,'r')
    lines=fd.readlines()
    t0=[int(i.split('"')[0].split(',')[0]) for i in lines]
    t45=[int(i.split('"')[0].split(',')[1]) for i in lines]
    tM=[int(i.split('"')[0].split(',')[2]) for i in lines]
    
    d0_=[i.split('"')[1].split('[')[1].split(']')[0] for i in lines]
    d0=[]
    d0_=[i.split(',') for i in d0_]
    for i in d0_:
        a=[]
        for j in i:
            a.append(float(j))
        d0.append(a)
        
    d45_=[i.split('"')[3].split('[')[1].split(']')[0] for i in lines]
    d45=[]
    d45_=[i.split(',') for i in d45_]
    for i in d45_:
        a=[]
        for j in i:
            a.append(float(j))
        d45.append(a)
        
    dM_=[i.split('"')[5].split('[')[1].split(']')[0] for i in lines]
    dM=[]
    dM_=[i.split(',') for i in dM_]
    for i in dM_:
        a=[]
        for j in i:
            a.append(float(j))
        dM.append(a)
    
    return (t0,t45,tM,d0,d45,dM)

def read_data_set(location,file,raw_max=False):
    os.chdir(location)
    fd=open(file,'r')
    lines=fd.readlines()
    pf=[int(i.split('"')[0].split(',')[0]) for i in lines]
    imgloc=[int(i.split('"')[0].split(',')[1]) for i in lines]
    subloc=[int(i.split('"')[0].split(',')[2]) for i in lines]
    m0=[float(i.split('"')[0].split(',')[3]) for i in lines]
    m45=[float(i.split('"')[0].split(',')[4]) for i in lines]
    mM=[float(i.split('"')[0].split(',')[5]) for i in lines]    
    tw=[float(i.split('"')[0].split(',')[6]) for i in lines]
    
    if raw_max==True:
        mM_raw=[float(i.split(',')[-1]) for i in lines] 
    
    d0=[]
    d45=[]
    rad=[]
    
    for i in lines:
        try:
            d0_=i.split('"')[1].split('[')[1].split(']')[0]
            d0_=d0_.split(',')
            a=[]
            for k in d0_:
                a.append(float(k))
            d0.append(a)
            
            d45_=i.split('"')[3].split('[')[1].split(']')[0]
            d45_=d45_.split(',')
            a=[]
            for k in d45_:
                a.append(float(k))
            d45.append(a)
                
            rad_=i.split('"')[5].split('[')[1].split(']')[0]
            rad_=rad_.split(',')
            a=[]
            for k in rad_:
                a.append(float(k))
            rad.append(a)
                
        except IndexError:
            d0.append(float(i.split('"')[0].split(',')[7]))
            d45.append(float(i.split('"')[0].split(',')[8]))
            rad.append(float(i.split('"')[0].split(',')[9]))
    
    if raw_max==False:
        return (pf,imgloc,subloc,m0,m45,mM,tw,d0,d45,rad)
    else:
        return (pf,imgloc,subloc,m0,m45,mM,tw,d0,d45,rad,mM_raw)
        

#SUBDIVIDE IMAGE INTO SET OF SMALLER IMAGES
def subsub(image1,xdim,ydim,dx,dy):
    '''
    SUBDIVIDE IMAGE INTO SET OF SMALLER IMAGES:
    This divides a image1 with dimensions (xdim by ydim pixels) into an array of smaller images, each with dimensions (dx by dy pixels)
    '''
    setofsmallerimages=[]
    for ii in range(0,ydim,dy):
        for jj in range(0,xdim,dx):
            setofsmallerimages.append(image1[ii:ii+dy,jj:jj+dx])
    return setofsmallerimages
    
def boxpoint(img,G,value,xdim=640,ydim=480,boxsize=20):
    '''
    This draws a box of 20 pixels lenght around pixel G
    
    value should be set to np.max(img) or np.min(img) such that the box is white or black
    '''
    X,Y=G%xdim,G//xdim
    for i in range(-int(boxsize/2),int(boxsize/2)+1):
        if 0<X+i and (X+i)<xdim and (Y+int(boxsize/2))<ydim:
            img[Y+int(boxsize/2)][X+i]=value
        if X+i<xdim and X+i>0 and Y-int(boxsize/2)>0:
            img[Y-int(boxsize/2)][X+i]=value
        if 0<Y+i and Y+i<ydim and X+int(boxsize/2)<xdim:
            img[Y+i][X+int(boxsize/2)]=value
        if 0<Y+i and Y+i<ydim and X-int(boxsize/2)>0:
            img[Y+i][X-int(boxsize/2)]=value
    return img
    

subsub.func_doc="Image,xdim,ydim,dx,dy: retruns subdivided image"

def lightmask(img,AveLight,StdLight,MeanAveLight,StdAveLight,MeanStdLight,StdStdLight):
    #x=0.5, y=0.05 too stringent
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    AA: MeanAveLight is the average of the average values of all of the light images from the wafer
    SA: StdAveLight is the STD of all of the mean values from light images
    
    AS: MeanStdLight is the mean value of the standard deviation of light level taken from each image
    SS: StdStdLight is the standard deviaton of the standard deviations taken from each light image    
    
    AveLight is the aveage light level of one specific image
    StdLight is the standard deviation of the light level of one specific image
    
    LOOKS AT IMG LIGHT LEVEL AND STANDARD DEVIATION WITH RESPECT TO ALL OTHER IMAGES 
    TO DETERMINE WHETHER THE IMAGE CONTAINS MASK OR NOT AND FLAG IT IF IT DOES
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    
    #Observe wafers Light Lists to Determine pass/fail
    lightmask=[]
    
    #AVE LIGHT MASK
    x=1.0
    if AveLight>(MeanAveLight+(x*StdAveLight)) or AveLight<(MeanAveLight-(0.66*x*StdAveLight)):
        lightmask.append(int(0))
    else:
        lightmask.append(int(1))
    
    #STD LIGHT MASK
    y=0.8
    if StdLight>(MeanStdLight+(y*StdStdLight)):
        lightmask.append(int(0))
    else:
        lightmask.append(int(1))
    
    #RETURN TRUE IF THE IMAGE PASSES THE LIGHT MASK FILTER
    if np.sum(lightmask)==2:
        return True
    else:
        return False

def bowscan(img0,img45,index):
    global d0,d45,t0,t45    
    
    d0,d45,t0,t45=0,0,0,0    
    
    Fit=True
        
    #GET NORMALIZED CIRCLE SCAN, TEST WITH CLASSIFIER AND APPEND TO CLASSIFICATION LIST
    theta_0=0
    for m in ['Shear 0','Shear 45']:           
        idict={'Shear 0':img0,'Shear 45':img45}
        rdict={'Shear 0':True,'Shear 45':False}
        
        img=idict[m]
        Rot=rdict[m]
        
        scan.xdim=img0.shape[1]
        scan.ydim=img0.shape[0]
        
       
        meanvals,meanvals_raw,thetas,thetamax=scan.classify_scan_fit_rot(img,index,theta_0,fit=Fit,rot=Rot)
        
        if m=='Shear 0':
            theta_0=thetamax
            
        
        globals()['t%s'%m[6:]]=int(thetamax*180/np.pi)
        globals()['d%s'%m[6:]]=meanvals
        
    features=[]
    if int(t0-t45)<0:
        features.append(int(t0-t45+180))
    else:
        features.append(t0-t45)
        
    for ii in d0:
        features.append(ii)
    for ii in d45:
        features.append(ii)
    
    return features
    
def sigperp(sig11,sig12,sig22,alpha):
    '''
    TAKES IN BULK RESIDUAL STRESS FIELD AND CRACK ORIENTATION.  RETURNS STRESS NORMAL TO CRACK.
    SEE MAPLE C:\Users\Logan Rowe\Desktop\Protected Folder\Image Processing\6 Calculate KI KII and Alpha
    '''
    sigperp=(np.cos(alpha)**2)*(sig22-sig11)+sig11+(2*np.cos(alpha)*np.sin(alpha)*sig12)
    return sigperp

if __name__=='__main__':
    plt.close('all')
    
    theta=np.linspace(0,2*np.pi,500)
    dM=[dM_calc_normalized(10,1,i) for i in theta]
    lv=np.min(dM)
    dM=[i-lv for i in dM]
    uv=np.max(dM)
    dM=[i/uv for i in dM]
    plt.figure(1)
    plt.plot(theta,dM)
    



    
    