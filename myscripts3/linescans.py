# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 15:36:38 2018

@author: Logan Rowe
"""

from __future__ import division
import numpy as np
import scipy as sp
import scipy.signal._savitzky_golay as sg
import matplotlib.pyplot as plt

xdim,ydim=640,480

def circlesweep(img,G,R,res,xdimension=640,theta_0=0,CCW=True):
    '''
    Performs a linescan from center of circle to edge (length R) and sweeps 360 degrees around the circle starting at angle theta_0
    
    G: index of pixel you wish to sweep around (int)
    R: radius in pixels of the line that will be swept in a circle (int)
    res: increment of line that will be swept (pixels)
    theta_0: starting angle of the line that will be swept theta_0=0 is along the 100 direction (radians)
    CCW: if True will sweep the line in a counter-clockwise motion, if False then clockwise motion
    '''
    X,Y=G%xdimension,G//xdimension
    
    meanvals=[]
    thetas=[]
    
    theta=theta_0    
    while theta<(theta_0+(2*np.pi)):
        
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
    
    return (meanvals,thetas)
    
def circlesweep_test(img,G,R,res,xdimension=640,theta_0=0,CCW=True):
    '''
    Check the direction that circle sweep goes CW or CCW and starting at x+ or x-
    This is just for testing purposes and will likely not be called upon in code
    
    G: index of pixel you wish to sweep around (int)
    R: radius in pixels of the line that will be swept in a circle (int)
    res: increment of line that will be swept (pixels)
    theta_0: starting angle of the line that will be swept theta_0=0 is along the 100 direction (radians)
    CCW: if True will sweep the line in a counter-clockwise motion, if False then clockwise motion

    '''
    
    X,Y=G%xdimension,G//xdimension
    
    meanvals=[]
    thetas=[]
    xlist=[]
    ylist=[]
        
    theta=theta_0    
    while theta<(theta_0+(2*np.pi)):
        
        vals=[] #CU
        
        r=0
        while r<=R:
            if CCW==True: #Change direction of sweep
                t=-theta
            else:
                t=theta
            x,y=r*np.cos(t),r*np.sin(t)
            try:
                vals.append(img[int(Y+y)][int(X+x)])
                r+=0.5
            except:
                r+=0.5
                pass
        xlist.append(x)
        ylist.append(y)
        
        meanvals.append(np.mean(vals))
        thetas.append(theta)
        
        theta+=float(2*np.pi/res)
    
    return (meanvals,thetas,xlist,ylist)
    



def linescan(img,G,R,theta):
    '''
    Linescan across a given pixel of length 2R (R is in pixels) at angle theta (theta is measured from the x+ axis aka the [1,0,0] direction on the wafer)
    Returns linescan (values, radius) and mean value
    
    img: image which you wish to perform linescan on (numpy array)
    G: index of pixel at the center of the line scan (int)
    R: half the length of the linescan (pixels)
    theta: angle of the linescan with respect to x+ direction (radians)
    '''
    X,Y=G%xdim,G//xdim
    vals=[] #CU
    radius=[] #Pixels
    
    r=-R
    count=0
    while r<=R:
        x,y=r*np.cos(theta),r*np.sin(theta)
        try:
            vals.append(img[int(Y+y)][int(X+x)])
            radius.append(r)
            r+=0.5
            count+=1
        except:
            r+=0.5
            pass
    
    meanval=np.mean(vals)
    
    return (vals,radius,meanval)
    


def circlescan(img,G,R,theta_0,res): 
    '''
    Take average of line scans as linescan rotates about a pixel
    Image, central pixel of scan, half length of line in pixels, starting theta, resolution
    returns circle scan (meanvals,thetas,angle at which max value was observed)
    
    G: index of pixel you wish to sweep around (int)
    R: radius in pixels of the line that will be swept in a circle (int)
    res: increment of line that will be swept (pixels)
    theta_0: starting angle of the line that will be swept theta_0=0 is along the 100 direction (radians)
    CCW: if True will sweep the line in a counter-clockwise motion, if False then clockwise motion
    '''
    
    meanvals=[]
    thetas=[]
    
    theta=theta_0
    while theta<=(theta_0+np.pi):
        V,rad,M=linescan(img,G,R,theta)
        meanvals.append(M)
        thetas.append(theta)
        theta+=float(np.pi/res)
    
    thetamax_=thetas[np.argmax(meanvals)]
    
    while len(meanvals)>res:
        meanvals.pop()
    
    while len(thetas)>len(meanvals):
        thetas.pop()
    
    return (meanvals,thetas,thetamax_)
    
def savgol(data,windowsize,power):
    '''
    Savitzky-Golay Filter
    
    data: 1D array or List of data points you wish to smooth (i.e. meanvals from circlescan() or circlesweep())
    windowsize: The length of the filter window (i.e. the number of coefficients)
    power: The order of the polynomial used to fit the samples. power must be less than windowsize.
    
    See scipy.signal.savgol_filter documentation for more dtails
    '''
    #newdata=sp.signal.savgol_filter(data,windowsize,power)
    newdata=sg.savgol_filter(data,windowsize,power)
    return newdata

def classify_scan(img,G):
    windowsize,power,res,R,theta_0=15,5,50,4,0
    #Window to perform savgol over, power to use (3 provides smooth fit), number of averaged linescans per 180deg, pixel radius to linescan over, starting theta (arbitrary since max theta will be determined and used)
    
    meanvals,thetas,thetamax=circlescan(img,G,R,theta_0,res)
    
    meanvals_fit=savgol(meanvals,windowsize,power)
    #Base thetamax on fit to ignore outlier spikes in data            
    thetamax=thetas[np.argmax(meanvals_fit)]
    
    #Change name of thetamax to thetamax_0 to not overwrite thetamax
    meanvals,thetas,thetamax_0=circlescan(img,G,R,thetamax,res)
    
    #Minimum value up to zero scale maximum value to 1
    meanvals_fit=savgol(meanvals,windowsize,power)
    
    lower=np.min(meanvals_fit)
    upper=np.max(meanvals_fit-lower)
    meanvals=[(ii-lower)/upper for ii in meanvals] 
    meanvals_fit=[(ii-lower)/upper for ii in meanvals_fit] 
    
    return (meanvals_fit,meanvals,thetas)
    
def classify_scan_fit(img,G,fit=True):
    '''
    Returns (meanvals_fit,meanvals,thetas,thetamax) if fit=True
    Returns (meanvals,thetas,thetamax) if fit=False
    
    
    Performs circle scan on image around pixel index G
    Normalizes the values of the scan to range from between 0 and 1
    If fit=True then the thetas vs meanvals curve will be smoothed by a Savitzky Golay Filter
    
    img: the image that contains pixel G (numpy array)
    G: the index of the pixel of interest (int)
    meanvals: a list of average linescans as the line scan is rotated around the pixel
    meanvals_fit: meanvals smoothed by a Sav-Gol filter
    '''
    
    windowsize,power,res,R,theta_0=51,7,100,4,0
    #Window to perform savgol over, power to use (3 provides smooth fit), number of averaged linescans per 180deg, pixel radius to linescan over, starting theta (arbitrary since max theta will be determined and used)
    
    scale=100 #for machine learning purposes
    
    if fit==True:
        meanvals,thetas,thetamax=circlescan(img,G,R,theta_0,res)
        
        meanvals_fit=savgol(meanvals,windowsize,power)
        #Base thetamax on fit to ignore outlier spikes in data            
        thetamax=thetas[np.argmax(meanvals_fit)]
        
        #Change name of thetamax to thetamax_0 to not overwrite thetamax
        meanvals,thetas,thetamax_0=circlescan(img,G,R,thetamax,res)
        
        #Minimum value up to zero scale maximum value to 1
        meanvals_fit=savgol(meanvals,windowsize,power)       
        
        lower=np.min(meanvals_fit)
        upper=np.max(meanvals_fit-lower)
        meanvals=[scale*(ii-lower)/upper for ii in meanvals] 
        meanvals_fit=[scale*(ii-lower)/upper for ii in meanvals_fit] 
        
        return (meanvals_fit,meanvals,thetas,thetamax)
    
    else:
        meanvals,thetas,thetamax=circlescan(img,G,R,theta_0,res)
        meanvals,thetas,thetamax_0=circlescan(img,G,R,thetamax,res)
        lower=np.min(meanvals)
        upper=np.max(meanvals-lower)
        meanvals=[scale*(ii-lower)/upper for ii in meanvals] 
        
        return (meanvals,thetas,thetamax)

def classify_scan_fit_rot(img,G,theta_0,fit=True,rot=False):
    windowsize,power,res,R=51,7,100,4
    #Window to perform savgol over, power to use (3 provides smooth fit), number of averaged linescans per 180deg, pixel radius to linescan over, starting theta (arbitrary since max theta will be determined and used)
    
    scale=100 #for machine learning purposes
    
    if fit==True and rot==True: #Sav-Gol Fit of scan, rotated to start scan at peak value
        meanvals,thetas,thetamax=circlescan(img,G,R,theta_0,res)
        
        meanvals_fit=savgol(meanvals,windowsize,power)
        #Base thetamax on fit to ignore outlier spikes in data            
        thetamax=thetas[np.argmax(meanvals_fit)]
        
        #Change name of thetamax to thetamax_0 to not overwrite thetamax
        meanvals,thetas,thetamax_0=circlescan(img,G,R,thetamax,res)
        
        #Minimum value up to zero scale maximum value to 100
        meanvals_fit=savgol(meanvals,windowsize,power)       
        
        lower=np.min(meanvals_fit)
        upper=np.max(meanvals_fit-lower)
        meanvals=[scale*(ii-lower)/upper for ii in meanvals] 
        meanvals_fit=[scale*(ii-lower)/upper for ii in meanvals_fit] 
        
        return (meanvals_fit,meanvals,thetas,thetamax)
    
    elif fit==False and rot==True: #Raw data (normalized) but rotated so starts at peak value
        meanvals,thetas,thetamax=circlescan(img,G,R,theta_0,res)
        meanvals,thetas,thetamax_0=circlescan(img,G,R,thetamax,res)
        lower=np.min(meanvals)
        upper=np.max(meanvals-lower)
        meanvals=[scale*(ii-lower)/upper for ii in meanvals] 
        
        return (meanvals,thetas,thetamax)
        
    elif fit==False and rot==False: #Raw data (normalized) not rotated to maximum value (starts scan at theta_0)
        meanvals,thetas,thetamax=circlescan(img,G,R,theta_0,res)
        lower=np.min(meanvals)
        upper=np.max(meanvals-lower)
        meanvals=[scale*(ii-lower)/upper for ii in meanvals] 
        
        return (meanvals,thetas,thetamax)        
    
    else: #Sav-Gol Fit of scan but not rotated so scan starts at theta_0
        meanvals,thetas,thetamax=circlescan(img,G,R,theta_0,res)
        meanvals_fit=savgol(meanvals,windowsize,power)
        
        #Base thetamax on fit to ignore outlier spikes in data            
        thetamax=thetas[np.argmax(meanvals_fit)]
        
        #Minimum value up to zero scale maximum value to 100
        lower=np.min(meanvals_fit)
        upper=np.max(meanvals_fit-lower)
        meanvals=[scale*(ii-lower)/upper for ii in meanvals] 
        meanvals_fit=[scale*(ii-lower)/upper for ii in meanvals_fit] 
        
        return (meanvals_fit,meanvals,thetas,thetamax)
    

if __name__=='__main__':
    print("Testing")
    img=np.full((480,640),255)
    G=0.5*480*640+320
    R=50
    M,V,X,Y=circlesweep_test(img,G,R,360,640,0)
    X=X[:-1]
    Y=Y[:-1]
    
    #TRUNCATE TO TEST DIRECTION
    trunc=60
    X=X[:-trunc]
    Y=Y[:-trunc]    
    
    for (i,j) in zip(X,Y):
        x,y=G%640,G//640
        img[int(y+j)][int(x+i)]=0
    
    plt.close('all')
    plt.imshow(img)
    
    