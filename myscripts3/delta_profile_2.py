# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 20:25:31 2018

@author: Logan Rowe
"""

from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.special as sp

def CU_to_retardation(CU,lambda_,theta_pk):
    delta=(lambda_*CU)/(2*np.pi*theta_pk)
    return delta #meters


def maple_to_py_converter(filename,folder,delimiter=' Maple'): #CONVERT MAPLE EQUATION INTO PYTHON EQUATION
    file=folder+'\\'+filename
    
    eqn=open(file).readlines()[0]
    
    #REPLACE ITEMS IN ITEM DICTIONARY
    items={'tan':'np.tan','sin':'np.sin','cos':'np.cos','^':'**','exp':'np.exp','BesselJ(0,':'sp.j0(','BesselJ(0.,':'sp.j0(','BesselJ(1,':'sp.j0(','BesselJ(1.,':'sp.j0('}
    items2={"arcnp.tan":"np.arctan","signum":"np.sign","Pi":"np.pi","hypergeom([1/4, 1/2],[3/2],-t**2/rho**2)":"sp.hyp2f1(0.25,0.5,1.5,-t**2/rho**2)"}
    for i in items:
        if i in eqn:
            eqn=eqn.replace(i,items[i])
            
    for i in items2:
        if i in eqn:
            eqn=eqn.replace(i,items2[i])
            
    #SAVE PYTHON READABLE FILE
    with open(file.split(delimiter)[:-1][0]+' Python.txt','w') as fd:
        fd.write(eqn)
    
    print()
    print(eqn)
    return eqn

def C(beta):  #Stress optic coefficient for normal stress oriented along beta
    p44=6.5e-13 #Pa-1
    p11_12=9.88e-13 #Pa-1
    n_0=3.54 #Ordinary refractive index
    C=(n_0**3/2)*(((np.sin(2*beta)/p44)**2+(np.cos(2*beta)/p11_12)**2)**(-0.5))
    return C

def KIC_psi(psi):
    KIC_110=1.21e6 #fracture toughness when tension applied along 110 direction for a 001 si wafer
    KIC_100=1.97e6#"" applied along 100 direction for a 001 si wafer
    KIC=(((np.sin(2*psi)/KIC_110)**2+(np.cos(2*psi)/KIC_100)**2)**(-0.5))
    return KIC

#direction of first principal residual stress (works for image input too)
def beta_bulk(d0,d45,C_0=C(0),C_45=C(np.pi/4)):
    del0=d0
    del45=d45
    beta=-np.arctan(-del45*C_45/del0/C_0+C_45*(del45**2/C_0**2+del0**2/C_45**2)**(1/2)/del0)
    beta=np.nan_to_num(beta)
    return beta
    
    

#Ratio of KI/KII based on stress field acting on crack tip, crack orientation
#phi assumed to be arccos(2/pi) because it varies from 0 to 1 over the course of phi=0..Pi/2 so the average value is as such
def K_Ratio(beta_bulk,alpha,eta,phi=np.arccos(2/np.pi)):
    beta=beta_bulk
    KR=(np.sin(-beta+alpha)**2+eta*np.cos(-beta+alpha)**2)/(1-eta)/np.cos(-beta+alpha)/np.sin(-beta+alpha)/np.cos(phi)
    return KR
    

#Direction of first principal stress near crack tip depends on direction of first principal stress of
#bulk residual stress acting on the crack (beta_bulk)
#eta is sig(1)/sig(2) ratio of principal bulk residual stresses
#alpha is crack tip orientation CCW from x+ axis of wafer
#KR is ratio of KI/KII
def beta_cracktip(alpha,beta_bulk,theta,eta):
    beta=beta_bulk
    KR=K_Ratio(beta_bulk,alpha,eta)
    
    beta_crack=np.arctan((KR*np.cos(2*alpha+1/2*theta)-KR*np.cos(2*alpha+5/2*theta)+np.sign(-np.cos(-beta+alpha)*np.sin(-beta+alpha))*(-2*KR**2*np.cos(2*theta)+2*KR**2+8*KR*np.sin(2*theta)+6*np.cos(2*theta)+10)**(1/2)+np.sin(2*alpha+5/2*theta)+3*np.sin(2*alpha+1/2*theta))/(KR*np.sin(2*alpha+5/2*theta)-KR*np.sin(2*alpha+1/2*theta)+3*np.cos(2*alpha+1/2*theta)+np.cos(2*alpha+5/2*theta)))
    return beta_crack
    

#NORMALIZED DELTA_M
#The retardation field (d_m) which arises due to a crack tip undergoing bulk residual stress is symmetric
#with respect to theta an only varies in magnitude and position but not shape wrt alpha
#if the effect that alpha has on C(beta) is not accounted for
#dM=C(beta)*dN this definition calculated dN(alpha,,beta_bulk,eta,theta,phi)
def dN(beta_bulk,alpha,theta,eta,phi=np.arccos(2/np.pi)):
    beta=beta_bulk
    dN=(-(np.sin(-beta+alpha)**2+eta*np.cos(-beta+alpha)**2)**2*np.cos(2*theta)+(np.sin(-beta+alpha)**2+eta*np.cos(-beta+alpha)**2)**2+4*(np.sin(-beta+alpha)**2+eta*np.cos(-beta+alpha)**2)*np.sin(2*theta)*(1-eta)*np.cos(-beta+alpha)*np.sin(-beta+alpha)*np.cos(phi)+(3*np.cos(2*theta)+5)*(1-eta)**2*np.cos(-beta+alpha)**2*np.sin(-beta+alpha)**2*np.cos(phi)**2)**(1/2)
    return dN
    
    
    
#THIS INCORPORATES C(B) but dM is normalized by all axisymmetric and constant factors
def dM(beta_bulk,alpha,theta,eta):
    dM=C(beta_cracktip(alpha,beta_bulk,theta,eta))*dN(beta_bulk,alpha,theta,eta)
    return dM
    

def read_circle_sweep_txt(file):
    filename=file
    fd=open(filename,'r')
    lines=fd.readlines()
    beta=[float(i.split('"')[0].split(',')[0]) for i in lines]
    eta=[float(i.split('"')[0].split(',')[1]) for i in lines]
    alpha=[float(i.split('"')[0].split(',')[2]) for i in lines]
    
    dM=[]
    
    for i in lines:
        dM_=i.split('"')[1].split('[')[1].split(']')[0]
        dM_=dM_.split(',')
        a=[]
        for k in dM_:
            a.append(float(k))
        dM.append(a)
        
    return (beta,eta,alpha,dM)
    
def imgM(img0,img45,C_0=C(0),C_45=C(np.pi/4)): #Works pixel value returns dM at pixel array returns image of dM
    beta=beta_bulk(img0,img45)
    C_B=C(beta)
    imgM=C_B*(((img0/C_45)**2+(img45/C_0)**2)**0.5)
    
    return imgM
    
'''~~~~AFTER CALCULATING alpha, beta, eta~~~'''

def S1_S2(eta,mM,m0,m45,thickness_saw_damage,lambda_,theta_pk):
    #Convert m0,m45 to retardation

    t_sd=thickness_saw_damage
    
    beta_loc=beta_bulk(m0,m45)
    C_B=C(beta_loc)
    
    dsig=mM/(C_B*t_sd)
    
    S1=(eta*dsig)/(eta-1)
    S2=dsig/(eta-1)
    
    return (S1,S2)

def chi(alpha,beta_bulk_,eta,S2,t,rho,theta_w,del_0,del_45,phi=np.arccos(2/np.pi)):
    del_M=imgM(del_0,del_45)    
    
    theta=theta_w
    beta_loc=beta_bulk(del_0,del_45) #at theta_w,rho
    C_B=C(beta_loc)
    
    beta=beta_bulk_ #based on mean 0 and mean 45
    
    chi=-2/3*4**(1/2)*(np.pi*(-3/4*(eta-1)**2*((np.cos(phi)**2+1/3)*np.cos(2*theta)+5/3*np.cos(phi)**2-1/3)*np.sin(-beta+alpha)**4+np.sin(2*theta)*np.cos(phi)*np.cos(-beta+alpha)*(eta-1)**2*np.sin(-beta+alpha)**3+3/4*(eta-1)*(((eta-1)*np.cos(phi)**2+2/3*eta)*np.cos(2*theta)+(-5/3+5/3*eta)*np.cos(phi)**2-2/3*eta)*np.sin(-beta+alpha)**2-np.sin(2*theta)*np.cos(-beta+alpha)*eta*np.cos(phi)*(eta-1)*np.sin(-beta+alpha)-1/4*eta**2*(np.cos(2*theta)-1))*rho)**(1/2)*del_M/t/S2/((eta-1)**2*((np.cos(phi)**2+1/3)*np.cos(2*theta)+5/3*np.cos(phi)**2-1/3)*np.sin(-beta+alpha)**4-4/3*np.sin(2*theta)*np.cos(phi)*np.cos(-beta+alpha)*(eta-1)**2*np.sin(-beta+alpha)**3-(eta-1)*(((eta-1)*np.cos(phi)**2+2/3*eta)*np.cos(2*theta)+(-5/3+5/3*eta)*np.cos(phi)**2-2/3*eta)*np.sin(-beta+alpha)**2+4/3*np.sin(2*theta)*np.cos(-beta+alpha)*eta*np.cos(phi)*(eta-1)*np.sin(-beta+alpha)+1/3*eta**2*(np.cos(2*theta)-1))/C_B/sp.hyp2f1(0.25,0.5,1.5,-t**2/rho**2)
    chi=np.abs(chi)
    
    return chi
    
def chi_KR(delM,del0,del45,dsig,m0,m45,alpha,beta,eta,theta_w,phi=np.arccos(2/np.pi),rho=2.5e-6,t=200e-6):
    beta_bulk_=beta_bulk(m0,m45)
    KR=K_Ratio(beta_bulk_,alpha,eta)    
    
    theta=theta_w
    beta_loc=beta_bulk(del0,del45) #at theta_w,rho
    C_B=C(beta_loc)
    
    chi=2*delM/C_B/t/sp.hyp2f1(0.25,0.5,1.5,-t**2/rho**2)/(-KR**2*np.cos(2*theta)+KR**2+4*KR*np.sin(2*theta)+3*np.cos(2*theta)+5)**(1/2)/abs(dsig*np.cos(-beta+alpha)*np.sin(-beta+alpha)*np.cos(phi))*np.pi**(1/2)*rho**(1/2)

    return chi

def KI_KII(chi,S2,eta,beta_bulk_,alpha,phi=np.arccos(2/np.pi)):
    beta=beta_bulk_
    
    KI=S2*((np.cos(beta-alpha+0.5*np.pi))**2+eta*(beta-alpha+0.5*np.pi)**2)*chi
    KII=S2*(1-eta)*np.sin(beta-alpha+0.5*np.pi)*np.cos(beta-alpha+0.5*np.pi)*np.cos(phi)*chi
    
    return (KI,KII)
    
def S_Perp(KI,Chi,alpha,psi=0,Sigma_applied=0):
    S_P=(KI/Chi)+Sigma_applied*np.sin(alpha-psi)**2
    return(S_P)
    







if __name__=='__main__':
    
    #EQUATION CONVERTER
    filename='chi Maple.txt'
    folder=r'C:\Users\Logan Rowe\Desktop\Protected Folder\Image Processing\FINAL KI KII ALPHA BY FIT 2'
    eqn=maple_to_py_converter(filename,folder)
    print()

    #TEST PLOT BETA CRACK TIP
    thetas=range(360)
    thetas=[i*np.pi/180 for i in thetas]
    #beta_cracktip(alpha,beta_bulk,theta,eta)
    betas=[beta_cracktip(-np.pi/4,0,i,2) for i in thetas]
    plt.close('all')
    plt.figure('beta crac')
    plt.plot(thetas,betas,'r-')
    
    #TEST PLOT BETA BULK
    d0_vals=np.linspace(-10,10,100)
    d45_vals=np.linspace(-10,10,100)
    betas0=[beta_bulk(i,1)*180/np.pi for i in d0_vals]
    betas45n=[beta_bulk(-1,i)*180/np.pi for i in d45_vals]
    betas45p=[beta_bulk(1,i)*180/np.pi for i in d45_vals]
    plt.figure('beta bulk')
    plt.plot(d0_vals,betas0,'r-',d45_vals,betas45n,'b-',d45_vals,betas45p,'g-')
    plt.legend(['d0','d45','d45'])
    
    #TEST C(BETA_CRACK_TIP)
    Cvals=[C(i) for i in betas]
    plt.figure('C(beta crack tip)')
    plt.polar(thetas,Cvals)
    
    #TEST dN
    dNvals=[dN(beta_bulk(3,1),np.pi/2,i,2) for i in thetas]
    plt.figure('dN')
    plt.plot(thetas,dNvals)
    
    #TEST dM (the normalized shear max retardation field)
    
    dMvals=[dM(beta_bulk(-1,1),0,i,-1) for i in thetas]
    plt.figure('dM')
    plt.plot(thetas,dMvals)   
    
    #FIND BOUNDS FOR ETA S.T. KR=-20..20
    alphas=np.concatenate([np.linspace(0.05,np.pi/2-0.05,400),np.linspace(np.pi/2+0.05,np.pi-0.05,400)],axis=0)
    KR=[K_Ratio(0,alpha,-1) for alpha in alphas]
    #KR2=[K_Ratio_2(0,alpha,-1) for alpha in alphas]
    plt.figure('KR(alpha)')
    plt.plot(alphas,KR)
    #plt.plot(alphas,KR2)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$K_R$')
    plt.title(r'$\beta$=0 and $\eta$=-1')
    
    #FIND BOUNDS FOR ETA S.T. KR=-20..20
    etas=np.concatenate([np.linspace(-10,0.9,400),np.linspace(1.1,10,400)])
    #etas=np.concatenate([np.linspace(1.1,2.1,50),np.linspace(2.1,5.1,50)[1:],np.linspace(5.1,15.1,20)[1:],np.linspace(15.1,35.1,20)[1:],np.linspace(35.1,49.1,7)])
    KR=[K_Ratio(0,np.pi/4,eta) for eta in etas]
    plt.figure('KR(eta)')
    plt.plot(etas,KR,'ro')
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$K_R$')
    
    
    #TEST CALCULATION OF CHI
    #X=chi(alpha,beta_bulk,eta,s2,t,rho,tw,delm,del0,del45)
    S2=-100e6
    b_b=beta_bulk(1e-9,-10e-9)
    X=chi(np.pi/4,b_b,-1,S2,200e-6,5e-6,1,40e-9,np.sqrt(40)*1e-9,np.sqrt(40)*2e-9)
    print(X)
    print(1.2/X)
    
    KI,KII=KI_KII(X,S2,-2,b_b,np.pi/4)
    
    print('('+str(round(KI*1e-6,2))+','+str(round(KII*1e-6,2))+r') [MPa m^{0.5}]')
    
    plt.figure("KIC")
    psi=np.linspace(0,2*np.pi,200)
    KIC_=[KIC_psi(i) for i in psi]
    plt.polar(psi,KIC_)
    
