# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:41:40 2015

@author: lprowe2
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:43:07 2015

@author: lprowe2
"""
from array import *
import numpy as np
import csv
import sys
#from dt1toData_LtMsk_PxFndMsk_ShMxav_Indx import BadIndices

xdim=640
ydim=480

BadIndices=[]

def ConvList(From,To):
    for i in From:
        To.append(i)

#This Script takes a dt1 string input and converts it into shear 0, shear 45 and Light images.
#Requires from array import *; numpy as np 
#Must be in the folder with the dt1 files.  

def formimg(filename):
    global img0,img45,imgL
    if filename=="?":
        print "This script returns imgL, img0 and img45 from a dt1 file input as a string."
    else:
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
        
def isindexbad(index):
    global Test_, BadIndices
    Test_=False
    for k in BadIndices:
        if int(k) == int(index):
            Test_=True
            break
        else:
            Test_=False

def resetbadindices():
    global BadIndices
    BadIndices=[]

def buildbadindexlist(indexwatcher):
    global BadPix_, BadIndices
    BadPix_=False
    q=0
    indexwatcher.sort()
    for i in indexwatcher:
        if i==indexwatcher[q-1] and i==indexwatcher[q-2] and i==indexwatcher[q-3] and not i==indexwatcher[q-4]:
            BadPix_=True
            BadIndices.append(i)
        q+=1    
        
            
def WriteRows(rows,filepath,name):
    csvfile = filepath + '\\' + name + '.txt'
    with open(csvfile, "w") as output:
        writer = csv.writer(output,lineterminator='\n')
        for row in rows:
           writer.writerow(row)
           
def WriteRow(row,filepath,name):
    csvfile = filepath + '\\' + name + '.txt'
    with open(csvfile, "w") as output:
        writer = csv.writer(output,lineterminator='\n')
        writer.writerow(row)

def rebin(a,shape):
    sh=shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)
    
def LocalToGlobalIdx(LocalIndex,SubSubImgIndex,Xpixelspersubsub,Ypixelspersubsub,Xsubsubimagesacross):
    globalidx_=(SubSubImgIndex//Xsubsubimagesacross)*Xpixelspersubsub*Ypixelspersubsub*Xsubsubimagesacross+(LocalIndex//Xpixelspersubsub)*(Xsubsubimagesacross*Xpixelspersubsub)+(SubSubImgIndex%Xsubsubimagesacross)*Xpixelspersubsub+LocalIndex%Xpixelspersubsub
    return globalidx_

def lookaround(image,index,distance,peakintensity,meanintensity):
    #Returns measured radius of inclusion in units of pixel widths (not diagonals)
    #Image if global x dim and y dim shoudl not be a sub sub image
    #index thus refers to global index
    #Distance is pixels over which to search for radius size (10) does the job usually
    #peak intensity is image[index//xdim,index%xdim] the value of the central pixel noted as a max in a sub or sub sub image
    global ydim,xdim
    ydim,xdim=480,640
    m=1
    k=0
    j=0
    while m<9:
        globals()['r%s'%m]=[]
        m+=1
    while k<=distance:
        try:
            r1.append(image[(index//xdim)+k,(index%xdim)])
            r2.append(image[(index//xdim)+k,(index%xdim)+k])
            r3.append(image[(index//xdim),(index%xdim)+k])
            r4.append(image[(index//xdim)-k,(index%xdim)+k])
            r5.append(image[(index//xdim)-k,(index%xdim)])
            r6.append(image[(index//xdim)-k,(index%xdim)-k])
            r7.append(image[(index//xdim),(index%xdim)-k])
            r8.append(image[(index//xdim)+k,(index%xdim)-k])
        except:
            pass
        k+=1

    #Remove mean value of image since we are only concerned with retardation due to inclusion if value<mean then value=0
    m=1
    while m<9:
        globals()['r%s'%m]=[q-meanintensity for q in globals()['r%s'%m]]
        qq=0
        for qqq in globals()['r%s'%m]:
            if abs(qqq)>qqq:
                globals()['r%s'%m][qq]=0
            qq+=1
        m+=1
        

    #Interpolate the radii to find at what pixel distance the intensity has dropped to 25% of the maximum    
    
    ydesired=0.25*peakintensity
    
    radii=[]
    
    bigfakeradius=400    
    
    k=1
    while k<9:
        j=0
        val=True
        while val==True:
            if globals()['r%s'%k][j]>ydesired:
                j+=1
                if j>=len(globals()['r%s'%k]):
                    radii.append(bigfakeradius)
                    val=False
            else:
                y0=globals()['r%s'%k][j-1]
                y1=globals()['r%s'%k][j]
                x0=j-1
                x1=j
                try:
                    r=float(x0*ydesired-x0*y1-x1*ydesired+x1*y0)/float(y0-y1)
                except:
                    r=0.5*(x0+x1)
                if k%2==0:
                    radii.append(np.sqrt(2)*(r-0.5))
                else:
                    radii.append(r-0.5)
                val=False
        k+=1
        
    radii.sort()
    
    while len(radii)>0 and radii[-1:][0]==bigfakeradius:
            radii.pop()        
    
    if len(radii)>0:
        AveR=np.mean(radii)
        MaxR=np.max(radii)
    else:
        AveR=bigfakeradius
        MaxR=bigfakeradius
    
    return (AveR,MaxR)
'''   
if __name__=="__main__":
    import sys
    myscripts(int(sys.argv[0]))
'''