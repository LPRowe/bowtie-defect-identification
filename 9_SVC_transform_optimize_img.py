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

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.externals import joblib

from scipy.stats import expon,reciprocal

import data_prep as dp

from imp import reload
reload(dp)

import matplotlib.pyplot as plt





################################################
# LOAD DATA
################################################
data_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowtie-training-data'
X_raw=np.load(data_dir+'\\std0_std45_sh0-arr_sh45-arr_bow-bool_train.npy')


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


c1=['std0','std45']
c2=['sh0_{0}'.format(str(i)) for i in range(64)]
c3=['sh45_{0}'.format(str(i)) for i in range(64)]
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

y_train=train['bowties']  
X_train=train.drop(columns='bowties')

pipeline=Pipeline([('Max_Pooling',dp.combine_theta_peaks(combine_and_remove=False)),
                   ('Imputer',SimpleImputer(strategy='mean')),
                   ('Scaler',StandardScaler()),
                   ])


X_train_trans=pipeline.fit_transform(X_train)

param_distribs={'kernel':['rbf'],
                'gamma':expon(scale=1.0),
                'C':reciprocal(2,200000),
                }


svm_classifier=SVC()
rand_search=RandomizedSearchCV(svm_classifier,param_distributions=param_distribs,n_iter=30,cv=5,scoring='f1',verbose=2,n_jobs=2,random_state=42,iid=True)
rand_search.fit(X_train_trans,y_train)

params=rand_search.best_params_

clf=SVC(C=params['C'],gamma=params['gamma'],kernel='rbf')
clf.fit(X_train_trans,y_train)

y_test=test['bowties']
X_test=test.drop(columns='bowties')
X_test=pipeline.fit_transform(X_test)

y_preds=clf.predict(X_test)

F_CV=rand_search.best_score_    
P,R,F=precision_score(y_test,y_preds),recall_score(y_test,y_preds),f1_score(y_test,y_preds)

res_reduct_performance[res]=(P,R,F,F_CV,params) #Precision, Recall, F1 (when applied to test set), F1_CV (measured during cross validation training across 150 trials, parameters used for both)


joblib.dump(res_reduct_performance,"C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\classifiers\\SVM_circlesweep_res_dict.pkl")
