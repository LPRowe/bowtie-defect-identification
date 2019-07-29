# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 22:00:13 2019

@author: Logan Rowe
"""

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class reduce_features_in_sweep(BaseEstimator,TransformerMixin):
    def __init__(self,reduced_circle_sweep_res=1,first_index_of_sweep_in_X=5,bowtie_identifier=True):
        self.reduced_res=reduced_circle_sweep_res #Reduce the resolution of the circle sweep by a factor of 1,2,3,4,6,8, or 9
        self.first_index=first_index_of_sweep_in_X #X[:,?] is the index that the bowtie circle sweep starts on
        self.bowtie_identifier=bowtie_identifier #Is the identifier (bowtie: 1, nonbowtie: 0) included in X or already split from X
    def fit(self,X,y=None):
        return X
    def transform(self,X):
        
        #Reduce the resolution of the circle sweep by a factor of self.reduced_res
        mask=[bool((i)%self.reduced_res) for i in range(144)] #Masks every 3rd (or reduce value) data point in the shear 0 and shear 45 sweeps
        mask=np.logical_not(mask).tolist()
        if self.bowtie_identifier:
            mask.append(True) #Always keep the bowtie identifier unless it was already split from X
        mask_thetas_stds=[True]*self.first_index
        mask=mask_thetas_stds+mask
    
        X=X[:,mask]
        return X

class combine_theta_peaks(BaseEstimator,TransformerMixin):
    def __init__(self,combine_and_remove=True):
        self.remove=combine_and_remove
    def fit(self,X,y=None):
        return X
    def transform(self,X):
        theta_0=X[:,1]
        theta_45=X[:,2]
        theta_abs=np.abs(theta_0-theta_45)
        if self.remove:
            X_=np.c_[X[:,0],theta_abs,X[:,3:]]
            return X_
        else:
            X_=np.c_[X[:,0:3],theta_abs,X[:,3:]]
            return X_