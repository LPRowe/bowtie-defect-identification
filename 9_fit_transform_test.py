# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:17:34 2019

@author: Logan Rowe
"""

import numpy as np
import os
import sys
import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.impute import SimpleImputer


from scipy.stats import expon,reciprocal

import data_prep as dp

from imp import reload





################################################
# LOAD DATA
################################################
data_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowtie-training-data'
X=np.load(data_dir+'\\thetaM_theta0_theta45_std0_std45_sh0_sh45_bow-bool_train_144_dim.npy')


################################################
# ADD COLUMN NAMES AND CONVERT TO PANDAS DF
################################################
def numpy_to_pd(array,column_names):
    '''
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

c1=['thetaM','theta0','theta45','std0','std45']
c2=['sh0_{0}'.format(str(i)) for i in range(72)]
c3=['sh45_{0}'.format(str(i)) for i in range(72)]
c4=['bowties']
column_names=c1+c2+c3+c4

X=numpy_to_pd(X,column_names)

################################################
# SPLIT DATA INTO TEST AND TRAIN BOTH BALANCED
# WITH RESPECT TO (NON)BOWTIES
################################################
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(X,X['bowties']):
    train=X.loc[train_index]
    test=X.loc[test_index]
    
print(test.shape)
    
X_=dp.reduce_features_in_sweep(first_index_of_sweep_in_X=list(X.columns).index('sh0_0')).transform(test)
X_=dp.combine_theta_peaks().transform(X_)


corr_matrix=X_.corr()
#corr_vals=corr_matrix['bowties'].sort_values(ascending=False)
#print(corr_vals)
print(X_.shape)

'''    
pipeline=Pipeline([('Imputer',SimpleImputer(strategy='mean')),
        ('Reduce Features',dp.reduce_features_in_sweep(reduced_circle_sweep_res=2, first_index_of_sweep_in_X=list(X.columns).index('sh0_0'))),
                   ])
'''



#Attribute Adder
##Generate new attribute absolute difference of shear0 and shear 45 max points
#Attribute Remover



'''
param_distribs={'kernel':['rbf'],
                'gamma':expon(scale=1.0),
                'C':reciprocal(20,200000),
                }

svm_reg=SVC()
'''