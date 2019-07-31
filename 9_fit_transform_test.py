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

from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from scipy.stats import expon,reciprocal

import data_prep as dp

from imp import reload
reload(dp)





################################################
# LOAD DATA
################################################
data_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowtie-training-data'
X_raw=np.load(data_dir+'\\thetaM_theta0_theta45_std0_std45_sh0_sh45_bow-bool_train_144_dim.npy')


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

X=numpy_to_pd(X_raw,column_names)

################################################
# SPLIT DATA INTO TEST AND TRAIN BOTH BALANCED
# WITH RESPECT TO (NON)BOWTIES
################################################
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(X,X['bowties']):
    train=X.loc[train_index]
    test=X.loc[test_index]

y=train['bowties']  
X=train.drop(columns='bowties')

'''
X=dp.reduce_features_in_sweep(first_index_of_sweep_in_X=list(X.columns).index('sh0_0'),
                              reduced_circle_sweep_res=4).transform(X)
X=dp.combine_theta_peaks(combine_and_remove=False).transform(X)


corr_matrix=X.corr()
corr_vals=corr_matrix['bowties'].sort_values(ascending=False)
print(corr_vals)
'''

'''
pipeline=Pipeline([('Imputer',SimpleImputer(strategy='mean')),
                   ('Reducer',dp.reduce_features_in_sweep(first_index_of_sweep_in_X=list(X.columns).index('sh0_0'),
                              reduced_circle_sweep_res=4)),
                   ('ThetaDiff',dp.combine_theta_peaks(combine_and_remove=False)),
                   ('Scaler',StandardScaler()),
                   ])
'''

pipeline=Pipeline([('Reducer',dp.reduce_features_in_sweep(first_index_of_sweep_in_X=list(X.columns).index('sh0_0'),
                              reduced_circle_sweep_res=4)),
                   ('ThetaDiff',dp.combine_theta_peaks(combine_and_remove=False)),
                   ('Imputer',SimpleImputer(strategy='mean')),
                   ('Scaler',StandardScaler()),
                   ])


X_trans=pipeline.fit_transform(X)

param_distribs={'kernel':['rbf'],
                'gamma':expon(scale=1.0),
                'C':reciprocal(20,200000),
                }


svm_classifier=SVC()
rand_search=RandomizedSearchCV(svm_classifier,param_distributions=param_distribs,n_iter=10,cv=5,scoring='neg_mean_squared_error',verbose=2,n_jobs=4,random_state=42)
rand_search.fit(X_trans,y)
print(rand_search.best_params_)
#'C': 329.6089285595793, 'gamma': 0.7439278308608545, 'kernel': 'rbf'
params=rand_search.best_params_

clf=SVC(C=params['C'],gamma=params['gamma'],kernel='rbf')
clf.fit(X_trans,y)

test_y=test['bowties']
test_x=test.drop(columns='bowties')
test_x=pipeline.fit_transform(test_x)

preds=clf.predict(test_x)

true_pos=0
false_pos=0
false_neg=0
for (i,j) in zip(preds,test_y):
    if i==j and i==1:
        true_pos+=1
    elif i!=j and i==1:
        false_pos+=1
    elif i!=j and i==0:
        false_neg+=1

precision=true_pos/(true_pos+false_pos)
recall=true_pos/(true_pos+false_neg)

print('P:',str(precision))
print('R:',str(recall))

