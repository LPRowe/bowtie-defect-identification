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

import skimage.measure

def prec_rec_f(predictions,actual):
    '''
    Takes predicted values from classifer and actual values from data set
    
    Returns: (precision_score, recall_score, f1_score)
    
    predictions=[1,1,0,0]
    actual=[1,0,0,0]
    P,R,F=prec_rec_f(predictions,actual)
    
        (P,R,F)
        >>>(1.0, 0.5, 0.6666666666666666)
    '''
    
    true_pos=0
    false_pos=0
    false_neg=0
    for (i,j) in zip(predictions,actual):
        if i==j and i==1:
            true_pos+=1
        elif i!=j and i==1:
            false_pos+=1
        elif i!=j and i==0:
            false_neg+=1

    precision=true_pos/(true_pos+false_pos)
    recall=true_pos/(true_pos+false_neg)
    f1=(2*precision*recall/(precision+recall))
    
    return (precision,recall,f1)


def numpy_to_pd(array,column_names):
    '''
    Converts a numpy array to a pandas dataframe given an array and list of column names.
    
    array: numpy array
    column_names: list of strings where each item is a column name
    
    Example:
        array=np.array([[1,2],[3,4]])
        column_names=['c_1','c_2']
        X=numpy_to_pd(array,column_names)
        print(X)
        >>>pandas.DataFrame(dict({'c_1':[1,3],'c_2':[2,4]}))
    '''
    pd_dict=dict()
    for (name,index) in zip(column_names,range(len(column_names))):
        pd_dict[name]=array[:,index]
    return pd.DataFrame(pd_dict)

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
        
        
class max_pooling(BaseEstimator,TransformerMixin):
    def __init__(self,pool_side_length=2,image_side_length=8):
        self.pool_side_length=pool_side_length
        self.image_side_length=image_side_length
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        if self.pool_side_length==1:
            #Effectively turns off max_pooling
            return X
        if self.pool_side_length==self.image_side_length:
            #Effectively highest pixel value in array
            
            #Maximum of sh0 values
            sh0_mask=['sh0' in i for i in X.columns.tolist()]
            X_sh0_max=X[X.columns[sh0_mask]].max(axis=1)

            #Maximum of sh45 values
            sh45_mask=['sh45' in i for i in X.columns.tolist()]
            X_sh45_max=X[X.columns[sh45_mask]].max(axis=1)
            
            return np.c_[X[['std0','std45']],X_sh0_max,X_sh45_max]
        
        #only columns that contain sh0 data
        sh0_mask=['sh0' in i for i in X.columns.tolist()]
        
        #only columns that contain sh45 data
        sh45_mask=['sh45' in i for i in X.columns.tolist()]

        #Make spearate pandas DF for shear 0 and shear 45 data
        X_sh0=X[X.columns[sh0_mask]]
        X_sh45=X[X.columns[sh45_mask]]
        
        #Convert to 1D numpy array
        X_sh0=np.array(X_sh0)
        X_sh45=np.array(X_sh45)
        
        block_reduced_0=[]
        block_reduced_45=[]
        for (i0,i45) in zip(X_sh0,X_sh45):
            #Reshape into the a list of 2D array images
            i0=np.reshape(i0,(self.image_side_length,self.image_side_length))
            i45=np.reshape(i45,(self.image_side_length,self.image_side_length))
        
            #MaxPool each 2 by 2 region
            i0=skimage.measure.block_reduce(i0,(self.pool_side_length,self.pool_side_length),np.max)
            i45=skimage.measure.block_reduce(i45,(self.pool_side_length,self.pool_side_length),np.max)  
            
            #Reshape back into a list
            i0=i0.reshape((1,-1))
            i45=i45.reshape((1,-1))
            
            #Add max_pooled images to lists to be added to array
            block_reduced_0.extend(i0)
            block_reduced_45.extend(i45)
        
        self.pixels_per_image=int((self.image_side_length/self.pool_side_length)**2)
        
        max_pooled_0=np.reshape(block_reduced_0,(-1,int(self.pixels_per_image)))
        max_pooled_45=np.reshape(block_reduced_45,(-1,int(self.pixels_per_image)))
        print(max_pooled_0.shape)
        
        
        #COMBINE MAX POOLED IMAGES WITH STANDARD DEVIATION SERIES AND RETURN NUMPY ARRAY
        sh_mask=np.array(sh0_mask)+np.array(sh45_mask)
        print(X[X.columns[~sh_mask]].shape)
        X_=np.c_[X[X.columns[~sh_mask]],max_pooled_0,max_pooled_45]
        
        return X_
          
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]        
        
        
def extract_features_XGBC(sh0_arr,sh45_arr,pixel):
    bs=8

    # =============================================================================
    # Crop array around bowtie
    # =============================================================================
    xdim,ydim=sh0.shape
    x_loc,y_loc=pixel%xdim,pixel//ydim
    sh0=sh0_arr[int(y_loc-0.5*bs):int(y_loc+0.5*bs),int(x_loc-0.5*bs):int(x_loc+0.5*bs)]
    sh45=sh45_arr[int(y_loc-0.5*bs):int(y_loc+0.5*bs),int(x_loc-0.5*bs):int(x_loc+0.5*bs)]
    
    # =============================================================================
    # Select features of interest and append them to a list
    # =============================================================================
    std0=np.std(sh0)
    std45=np.std(sh45)
    sh0=np.reshape(sh0,(1,-1))
    sh45=np.reshape(sh45,(1,-1))

    features=[std0,std45]
    features.extend(sh0)
    features.extend(sh45)
    
    return features
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
if __name__ == '__main__':
    print(prec_rec_f([1,0,0,0],[1,1,0,0]))
    