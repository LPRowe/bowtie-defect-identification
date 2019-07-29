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

from scipy.stats import expon,reciprocal





################################################
# LOAD DATA
################################################
data_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowtie-training-data'
X=np.load(data_dir+'\\thetaM_theta0_theta45_std0_std45_sh0_sh45_bow-bool_train_144_dim.npy')

c1=['thetaM','theta0','theta45','std0','std45']
c2=['sh0_{0}'.format(str(i)) for i in range(72)]
c3=['sh0_{0}'.format(str(i)) for i in range(72)]
c4=['bowtie']

column_names=c1+c2+c3+c4

X=pd.DataFrame(X)

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(X,X[:,-1]):
    train_set=X.loc[train_index]
    test_set=X.loc[test_index]

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