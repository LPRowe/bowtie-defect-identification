# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 22:00:13 2019

@author: Logan Rowe
"""

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class reduce_features_in_sweep(BaseEstimator,TransformerMixin):
    def __init__(self,reduced_circle_sweep_res=2,first_index_of_sweep_in_X=5,bowtie_identifier=False):
        self.reduced_res=reduced_circle_sweep_res #Reduce the resolution of the circle sweep by a factor of 1,2,3,4,6,8, or 9
        self.first_index=first_index_of_sweep_in_X #X[:,?] is the index that the bowtie circle sweep starts on
        self.bowtie_identifier=bowtie_identifier #Is the identifier (bowtie: 1, nonbowtie: 0) included in X or already split from X
    def fit(self,X,y=None):
        return self
    def transform(self,X, y=None):
        
        #Reduce the resolution of the circle sweep by a factor of self.reduced_res
        mask=[bool((i)%self.reduced_res) for i in range(144)] #Masks every 3rd (or reduce value) data point in the shear 0 and shear 45 sweeps
        mask=np.logical_not(mask).tolist()
        if self.bowtie_identifier:
            mask.append(True) #Always keep the bowtie identifier unless it was already split from X
        mask_thetas_stds=[True]*self.first_index
        mask=mask_thetas_stds+mask
    
        X_=X[X.columns[mask]]
        return X_

     
class combine_theta_peaks(BaseEstimator,TransformerMixin):
    def __init__(self,combine_and_remove=False):
        self.remove=combine_and_remove
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        theta_abs=[abs(i) for i in (X['theta0']-X['theta45']).tolist()]
        #theta_abs=pd.DataFrame(dict({'thetaDiff':theta_abs}))
        if self.remove:
            X['thetaDiff']=theta_abs
            X=X.drop(['theta0','theta45'],axis=1)
            return X
        else:
            X['thetaDiff']=theta_abs
            return X