# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:11:26 2018

@author: Logan Rowe
"""

from __future__ import division
from array import *
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import csv
import sys

def formimg(filename,filelocation):
    '''
    filename: .dt1 file that you wish to convert to a numpy array
    filelocation: the folder the file is located in
    
    READ A DT1 FILE AND RETURN 3 IMAGES: SHEAR 0, SHEAR 45 AND LIGHT IMAGES
    
    .dt1 files are structured such that they contain a header and 3 sequential images
    
    Reads the file backwards to detect @**@Data then splits the header from 
    the data at the index of @**@Data + 34
    
    Each image contains 640*480=307,200 data points (the number of pixels)
    The first 307,200 data points make up the IR-Transmission (Light) image
    the second set of data points make up the Shear 45 (img45) image
    and the third set of data points make up the Shear 0 (img0) image
    
    Since the raw Shear 45 and Shear 0 images must be divided by the light image
    to obtain the desired shear 45 and shear 0 measurements a singularity occurs
    when a pixel registers 0 for the light image.  Here this issue is addressed in
    img45_[img45_==np.inf]=1, however this issue may also be resolved by setting all 
    zero value Light image pixels to the average of the light image prior to forming
    the shear 0 and shear 45 images.
    '''
    
    
    os.chdir(filelocation)
    
    #These are the x and y dimensions of each image in pixels
    xdim,ydim=640,480 
    
    #reads the dt1 file backwards to find string preceeding image data
    filestring = open(filename,"rb").read()
    idx = filestring.rfind(b'@**@Data')
    
    #Separates data from header and parses it
    splitstring=filestring[idx+34:]
    f=array('f')
    f.fromstring(splitstring[:])
    f=np.asarray(f)

    #Reshape the IR Transmission (Light) image first as other images are normalized by it
    imgL = f[:xdim*ydim].reshape(ydim,xdim)

    #Reshape Shear 45 Image
    img45 = f[xdim*ydim:2*xdim*ydim].reshape(ydim,xdim)
    
    #Normalize Shear 45 Image by the Light Image
    #There is nothing special about the value 1 other than it is very low compared to other measured values
    #it is used here to handle infinite values in the shear 0 and shear 45 images that arise from an erroneous pixel value of 0 in a light image
    with np.errstate(divide='ignore', invalid='ignore'):
        img45_ = np.true_divide(img45,imgL)
        img45_[img45_ == np.inf] = 1
        img45_ = np.nan_to_num(img45_)  
        
    img45=img45_
    
    #Reshape the Shear 0 Image    
    img0 = f[2*xdim*ydim:3*xdim*ydim].reshape(ydim,xdim)     
    
    #Normalize Shear 0 Image by Light Image    
    with np.errstate(divide='ignore', invalid='ignore'):
        img0_ = np.true_divide(img0,imgL)
        img0_[img0_ == np.inf] = 1
        img0_ = np.nan_to_num(img0_)  
    
    img0=img0_
    
    
    return img0,img45,imgL


def C(theta): #Units Inverse Pascals
    """
    STRESS OPTIC COEFFICIENT FOR (001) SILICON
    Units are in inverse Pascals
    
    theta=0 along the 100 direction
    """
    p44=6.5e-13 #Pa-1
    p11_12=9.88e-13 #Pa-1
    n_0=3.54 #Ordinary refractive index
    C=(n_0**3/2)*(((np.sin(2*theta)/p44)**2+(np.cos(2*theta)/p11_12)**2)**(-0.5))
    return C

def beta(d0,d45): #Units: Radians
    """
    Takes shear 0 (d0) and shear 45 (d45) measurement and returns the direction of first principal stress.
    d0 and d45 can be floats, ints or numpy arrays
    
    Calculates the direction of the first principal stress accounting for anisotropy of stress optic coefficient
    Units are in Radians
    """
    C_45=C(np.pi/4)
    C_0=C(0)
    A=(d45*C_45)/(d0*C_0)
    beta=np.arctan(A-((1+A**2)**0.5))
    return beta

def shear_max_img(d0,d45): #Units: the same as d0 and d45
    """
    CALCULATES THE SHEAR MAX IMAGE FROM SHEAR 0 AND SHEAR 45 IMAGES
    """
    Beta=beta(d0,d45)
    C_0=C(0)
    C_B=C(Beta)
    C_45=C(np.pi/4)
    dM=C_B*((d45/C_0)**2+(d0/C_45)**2)**0.5
    return dM
    
def lightmask(AveLight,StdLight,MeanAveLight,StdAveLight,MeanStdLight,StdStdLight):
    #x=0.5, y=0.05 too stringent
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The purpose of hte light mask is to compare the light image of a single image to the rest of the images on the wafer
    if the image is too dark, too bright or too staticy then the light mask returns False signifying a bad image
    if the light mask returns True then the image is good

    The values x=1.0 and y=0.8 and 0.66*x have been optimized to mask the etched wafers but these values work reasonably well
    for as-cut wafers too    
    
    AA: MeanAveLight is the average of the average values of all of the light images from the wafer
    SA: StdAveLight is the STD of all of the mean values from light images
    
    AS: MeanStdLight is the mean value of the standard deviation of light level taken from each image
    SS: StdStdLight is the standard deviaton of the standard deviations taken from each light image    
    
    AveLight is the aveage light level of one specific image
    StdLight is the standard deviation of the light level of one specific image
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    #LOOKS AT IMG LIGHT LEVEL AND STANDARD DEVIATION WITH RESPECT TO ALL OTHER IMAGES 
    #TO DETERMINE WHETHER THE IMAGE CONTAINS MASK OR NOT AND FLAG IT IF IT DOES
    
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
        
def WriteRows(rows,filepath,name,tab=False):
    '''
    Write rows takes in
    
    rows: zip(list1,list2,list3,...)
    filepath: where the file should be saved
    name: the name of the file containing the zipped lists (no .txt in file name)
    
    and returns a comma delimited file where each row has one element from each list and there are N rows where N is the lenght of each list
    '''    
    
    csvfile = filepath + '\\' + name + '.txt'
    with open(csvfile, "w") as output:
        if tab!=True:
            writer = csv.writer(output,lineterminator='\n')
        else:
            writer = csv.writer(output,delimiter='\t',lineterminator='\n')
        for row in rows:
           writer.writerow(row)
           
           

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

"""   
def boxpoint(img,G,value,xdim=640,ydim=480,boxsize=20):
    '''
    This draws a box of 20 pixels lenght around pixel index G
    
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
"""

def boxpoint(img,G,value,xdim=640,ydim=480,boxsize=20):
    '''
    This draws a box of 20 pixels lenght around pixel index G
    
    value should be set to np.max(img) or np.min(img) such that the box is white or black
    '''
    X,Y=G%xdim,G//xdim
    
    #X=635
    #Y=200
    box_pixels=[]
    box_pixels.extend([(int(X-0.5*boxsize),int(j)) for j in range(int(Y-0.5*boxsize),int(Y+0.5*boxsize))])
    box_pixels.extend([(int(X+0.5*boxsize),int(j)) for j in range(int(Y-0.5*boxsize),int(Y+0.5*boxsize))])
    box_pixels.extend([(int(i),int(Y-0.5*boxsize)) for i in range(int(X-0.5*boxsize),int(X+0.5*boxsize))])
    box_pixels.extend([(int(i),int(Y+0.5*boxsize)) for i in range(int(X-0.5*boxsize),int(X+0.5*boxsize))])
    
    #Remove any off-image coordinates
    bpx=np.array([coord[0] for coord in box_pixels])
    bpy=np.array([coord[1] for coord in box_pixels])
    
    #mask true only for valid pixels
    mask=(bpx<=xdim-1)*(bpx>=0)*(bpy<=ydim-1)*(bpy>=0)
    
    masked_pixels=[]
    for (i,j) in zip(mask,box_pixels):
        if i:
            masked_pixels.append(j)
    
    for coord in masked_pixels:
        img[coord[1]][coord[0]]=value

    return img

def resetpixels(img,badpix,value,xdim=640):
    """
    img: np array of image
    badpix: list of hypersensitive pixel indices
    value: np.mean(image) or mean value of pixels neighboring the hot pixel (recommended)
    
    SET ALL HYPERSENSITIVE PIXELS TO EQUAL THE MEAN VALUE OF THE IMAGE
    THIS CAN EASILY BE ADAPTED TO MAKE THE VALUE THE AVERAGE OF THE PIXELS 4 CLOSEST NEIGHBORS (NOT A NECESSITY FOR MACHINE LEARNING PURPOSES)
    """
    for pix in badpix:
        img[int(pix)//xdim][int(pix)%xdim]=value
    return img
    
def circlesweep(img,G,R,res,xdim=640,theta_naught=0,CCW=True):
    """
    Performs a linescan from center of circle to edge (length R) 
    and sweeps 360 degrees around the circle starting at angle theta_0
    
    img: image sweep will be done in
    G: pixel index about which sweep will be done G=ylocation*xdim+xlocation
    R: radius of line scan (recommended 4 to capture most bowties)
    res: how many data points to collect (i.e. 100 will rotate the line scan by 3.6 degrees each time or 200 by 1.8 degrees each time)
    theta_naught: 
    CCW: counter clockwise linescan rotation if boolean=True
        
    returns (meanvals,thetas)
    """
    X,Y=G%xdim,G//xdim
    
    meanvals=[]
    thetas=[]
    
    theta=theta_naught    
    while theta<(float(theta_naught)+float(2*np.pi)):
        
        vals=[] #CU
        
        r=0
        while r<=R:
            if CCW==True: #Change direction of circle sweep CCW IS ACTUALLY CW ON IMAGE BUT CCW ON WAFER BECAUSE IMAGE IF LIPPED UP DOWN WHEN FORMED
                t=theta
            else:
                t=-theta
                
            x,y=r*np.cos(t),r*np.sin(t)
            try:
                vals.append(img[int(Y+y)][int(X+x)])
                r+=0.5
            except:
                r+=0.5
                pass
        
        meanvals.append(np.mean(vals))
        thetas.append(theta)
        
        theta+=float(2*np.pi/res)
    
    return (meanvals[:-1],thetas[:-1])

    

if __name__=='__main__':
    '''~~~~~~~~~FORM A SHEAR 0 SHEAR 45 AND LIGHT IMAGE~~~~~~~~~~~~~'''
    #SHEAR 0: SHOWS THE DIFFERENCE IN NORMAL STRAINS ALONG THE +/- 45 DEGREE DIRECTION
    #SHEAR 45: SHOWS THE DIFFERENCE IN NORMAL STRAINS ALONG THE X-Y DIRECTIONS
    #SHEAR MAX: SHOWS THE DIFFERENCE IN PRINCIPAL STRAINS
    #NOTE THESE ARE ALL IN CAMERA UNITS AND VALUES WOULD NEED TO BE CONVERTED TO STRESS OR STRAIN 
    #CONVERSION OF UNITS IS NECESSARY IF USING DATA FOR FRACTURE MECHANICS BUT NOT NECESSARY IF USING DATA FOR MACHINE LEARNING
    
    print("Forming First Set of Images")
    filelocation=r'E:\cSi Wafer Data\DeltaVision Scripts\Tutorial Images'
    os.chdir(filelocation)
    files=glob.glob('*.dt1')
    N=35 #image 10 of 40 in the test images
    img0,img45,imgL=formimg(files[N],filelocation) #There are 40 test images
    
    
    ydim,xdim=imgL.shape
    
    plt.close('all')
    
    plt.figure('Light')
    plt.gray()
    plt.imshow(imgL)
    plt.colorbar()
    
    plt.figure('Shear 0')
    plt.imshow(img0)
    plt.colorbar()
    
    plt.figure('Shear 45')
    plt.imshow(img45)
    plt.colorbar()
    
    
    '''~~~~~~~~~~~~~~~~~~~FORM SUBTRACTION IMAGE~~~~~~~~~~~~~~~~~~~~'''
    #A subtraction image contains measured retardation values that are the same in every image
    #By removing a subtraction image from the raw image we are removing aberrations caused by the detector and optical elements
    
    sub0=np.full((ydim,xdim),0)
    sub45=np.full((ydim,xdim),0)
    print("Forming Subtraction Image")
    for file in files:
        img0_,img45_,imgL_=formimg(file,filelocation)
        sub0+=img0_
        sub45+=img45_
    sub0=sub0/len(files)
    sub45=sub45/len(files)
    
    plt.figure("Subtraction Image 0")
    plt.imshow(sub0)
    plt.colorbar()
    
    plt.figure("Subtraction Image 45")
    plt.imshow(sub45)
    plt.colorbar()
    
    
    '''~~~~~~~~~~~~~~~~~~~APPLY SUBTRACTION IMAGE~~~~~~~~~~~~~~~~~'''
    print("Applying Subtraction Image")
    img0_raw=np.copy(img0) #These are the original img0 and img45 created above
    img45_raw=np.copy(img45)
    
    img0-=sub0 #These are the shear 0 and shear 45 images after removing the subtraction image
    img45-=sub45
    
    fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,sharex='col',sharey='row')
    fig.canvas.set_window_title("Compare Raw and Post Processes Shear Images")
    ax1.imshow(img0_raw)
    ax2.imshow(img45_raw)
    ax3.imshow(img0)
    ax4.imshow(img45)
    ax1.set_title(r'$\delta_0^{(raw)}$')
    ax2.set_title(r'$\delta_{45}^{(raw)}$')
    ax3.set_title(r'$\delta_{0}$=$\delta_0^{(raw)}$-$\delta_0^{(sub)}$')
    ax4.set_title(r'$\delta_{45}$=$\delta_{45}^{(raw)}$-$\delta_{45}^{(sub)}$')
    
    
    '''~~~~~~~~~~~~~~~~~~~~~~GENERATE THE SHEAR MAX IMAGE~~~~~~~~~~~~~~~~'''
    print("Generating Shear Max Image")
    #THE SHEAR MAX IMAGE IS RELATED TO THE DIFFERENCE IN PRINCIPAL STRESSES
    #ALL OF THE VALUES ARE STILL IN "CAMERA UNITS" ALTHOUGH FOR THE PURPOSE OF IDENTIFYING BOWTIES VIA MACHINE LEARNING
    #YOU NEVER NEED TO CONVERT INTO UNITS OF STRESS (PASCALS), FEEL FREE TO ASK ME IF YOU WANT TO LEARN HOW TO CONVERT THE UNITS
    
    imgM=shear_max_img(img0,img45)
    plt.figure("Shear Max")
    plt.imshow(imgM)
    plt.colorbar()
    
    '''~~~~~~~~~~~~~~~~~~~~~~SUBDIVIDE IMAGE AND FIND PEAK PIXELS~~~~~~~~~~~~~~~~'''
    #IF IMAGE DOES PASS SUBDIVIDE IMAGE AND GET LOCATION OF MAX SHEAR MAX
    dx,dy=160,120 #To make 4 images per 5x shear max image
    subimages=subsub(imgM,xdim,ydim,dx,dy) 

    #RECORD LOCATIONS OF PEAK RETARDATIONS
    localmaxes=[np.argmax(k) for k in subimages] #Returns list of peak pixels (one per sub image)
    maxidx=[LocalToGlobalIdx(k,m,dx,dy,xdim//dx) for (k,m) in zip(localmaxes,range(0,len(localmaxes)))] #Converts sub image pixel to full image pixel
    
    val=np.max(imgM)
        
    loccount=0 #Starting at sub image location 0
    for k in maxidx:        
       
        #This places a white box around each pixel of interest
        img0_boxed=boxpoint(img0,k,val)
        img45_boxed=boxpoint(img45,k,val)

        loccount+=1
        continue
    
    
    plt.figure("Boxed Shear 0")
    plt.imshow(img0_boxed)
    
    plt.figure("Boxed Shear 45")
    plt.imshow(img45_boxed)
    
    sublocation_of_interest=11
    
    #On the shear 0 image (img0) perform a line scan and sweep it in a circle around pixel location maxidx[0], where the line scan has a radius of 4 pixels, starting at angle theta=0 (theta_0=0), in a Counter Clockwise (CCW) direction
    pixel_values_0,theta=circlesweep(img0_boxed,maxidx[sublocation_of_interest],4,200,xdimension=640,theta_0=0,CCW=True) 
    
    #do the same for shear 45
    pixel_values_45,theta=circlesweep(img45_boxed,maxidx[sublocation_of_interest],4,200,xdimension=640,theta_0=0,CCW=True) 
    
    #SCALE SHEAR 0 AND SHEAR 45 TO BE FROM 0 TO 1
    pixel_values_0=[(i-np.min(pixel_values_0))/(np.max(pixel_values_0)-np.min(pixel_values_0)) for i in pixel_values_0]
    pixel_values_45=[(i-np.min(pixel_values_45))/(np.max(pixel_values_45)-np.min(pixel_values_45)) for i in pixel_values_45]


    plt.figure("Circle Scans Around Tutorial Bowtie")
    plt.plot(theta,pixel_values_0,'r-',theta,pixel_values_45,'b-')
    plt.legend(['Shear 0','Shear 45'])
    
    
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NOTE: 
    These scans are from the tutorial set which consists of only 40 images so the quality of the linescans shown is not going to be great
    
    Below is an example circle scan from the manually identified bowtie data set
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

    
    os.chdir('E:\\cSi Wafer Data\\DeltaVision Scripts\\Wafer Images\\Manually Identified Bowties_Given')
    bowtie_images=glob.glob('*0.npy')
    bowtie_of_interest=2
    
    bowtie_0=np.load(bowtie_images[bowtie_of_interest]) #load the shear 0 bowtie image (40 by 40 array)
    bowtie_45=np.load(bowtie_images[bowtie_of_interest][:-5]+'45.npy') #load the matching shear 45 bowtie image
    
    fig,((ax1,ax2))=plt.subplots(2,1)
    fig.canvas.set_window_title("Manually Identified Bowtie")
    ax1.imshow(bowtie_0)
    ax2.imshow(bowtie_45)
    ax1.set_title(r'Shear 0 Bowtie')
    ax2.set_title(r'Shear 45 Bowtie')
    
    imgM=shear_max_img(bowtie_0,bowtie_45)
    maxidx=np.argmax(imgM)
    
    #On the shear 0 image (img0) perform a line scan and sweep it in a circle around pixel location maxidx[0], where the line scan has a radius of 4 pixels, starting at angle theta=0 (theta_0=0), in a Counter Clockwise (CCW) direction
    pixel_values_0,theta=circlesweep(bowtie_0,maxidx,4,200,xdimension=40,theta_0=0,CCW=True) 
    
    #do the same for shear 45
    pixel_values_45,theta=circlesweep(bowtie_45,maxidx,4,200,xdimension=40,theta_0=0,CCW=True) 
    
    #SCALE SHEAR 0 AND SHEAR 45 TO BE FROM 0 TO 1
    pixel_values_0=[(i-np.min(pixel_values_0))/(np.max(pixel_values_0)-np.min(pixel_values_0)) for i in pixel_values_0]
    pixel_values_45=[(i-np.min(pixel_values_45))/(np.max(pixel_values_45)-np.min(pixel_values_45)) for i in pixel_values_45]


    plt.figure("Circle Scans Around Manually Identified Bowtie")
    plt.plot(theta,pixel_values_0,'r-',theta,pixel_values_45,'b-')
    plt.legend(['Shear 0','Shear 45'])
    