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

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
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

c1=['std0','std45']
c2=['sh0_{0}'.format(str(i)) for i in range(64)]
c3=['sh45_{0}'.format(str(i)) for i in range(64)]
c4=['bowties']
column_names=c1+c2+c3+c4

seeking=True
if seeking:
    X=dp.numpy_to_pd(X_raw,column_names)
    
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
    
    pipeline=Pipeline([('Imputer',SimpleImputer(strategy='mean'))])
    
    
    X_train_trans=pipeline.fit_transform(X_train)
    
    
    param_grid={'max_depth':[20],
                'bootstrap':[False],
                'criterion':['entropy'],
                'min_samples_leaf':[3],
                'max_features':['log2']
                }
    
    
        
    et_clf=ExtraTreesClassifier(n_estimators=10,n_jobs=-1,max_features='log2',random_state=42)
    grid_search=GridSearchCV(et_clf,param_grid=param_grid,cv=5,scoring='f1',verbose=2,n_jobs=-1,iid=True)
    grid_search.fit(X_train_trans,y_train)
    
    params=grid_search.best_params_
    
    clf=ExtraTreesClassifier(**params,n_estimators=1000,n_jobs=-1,random_state=42)
    clf.fit(X_train_trans,y_train)
    
    y_test=test['bowties']
    X_test=test.drop(columns='bowties')
    X_test=pipeline.fit_transform(X_test)
    
    y_preds=clf.predict(X_test)
    
    F_CV=grid_search.best_score_    
    P,R,F=precision_score(y_test,y_preds),recall_score(y_test,y_preds),f1_score(y_test,y_preds)

    print(P,R,F,F_CV,params)
    #0.8291457286432161 0.8549222797927462 0.8418367346938775 0.8303184083190176 {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 3}
    
final_params_selected=True
if final_params_selected:
    joblib.dump(clf,"C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\classifiers\\ET_img_classifier.pkl")

export_full_transformed_dataset=False
if export_full_transformed_dataset:
    processed_data_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\preprocessed_datasets'
    
    #Training Data Set
    training_full=np.c_[X_train_trans,np.array(y_train)]
    joblib.dump(training_full,processed_data_dir+'\\ETC_img_train.pkl')
    
    #Testing Data Set
    testing_full=test
    joblib.dump(testing_full,processed_data_dir+'\\ETC_img_test.pkl')