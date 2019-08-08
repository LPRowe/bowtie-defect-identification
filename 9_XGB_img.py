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

from sklearn.ensemble import RandomForestClassifier
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

import xgboost

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
    
    # =============================================================================
    #     SPLIT DATA INTO TEST AND TRAIN BOTH BALANCED
    #     WITH RESPECT TO (NON)BOWTIES
    # =============================================================================
    
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index, test_index in split.split(X,X['bowties']):
        train=X.loc[train_index]
        test=X.loc[test_index]
    
    # =========================================================================
    #     Split the training set into training and validation subsets
    # =========================================================================
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index, test_index in split.split(train,train['bowties']):
        train=X.loc[train_index]
        train_val=X.loc[test_index]
    
    y_train=train['bowties']  
    X_train=train.drop(columns='bowties')
    
    y_val=train_val['bowties']
    X_val=train_val.drop(columns='bowties')
    
    pipeline=Pipeline([('Imputer',SimpleImputer(strategy='mean'))])
    
    
    X_train_trans=pipeline.fit_transform(X_train)
    
    
    param_grid={'max_depth': [7],
     'learning_rate': [0.05],
     'n_estimators': [100],
     'verbosity': [1],
     'silent': [None],
     'objective': ['binary:logistic'],
     'gamma': [0],
     'min_child_weight': [5],
     'max_delta_step': [0],
     'subsample': [0.8],
     'colsample_bytree': [1],
     'colsample_bylevel': [1],
     'colsample_bynode': [0.2],
     'reg_alpha': [0],
     'reg_lambda': [1],
     'scale_pos_weight': [1],
     'base_score': [0.5],
     '_Booster': [None],
     'random_state': [42],
     'nthread': [None],
     'n_jobs': [1],
     'importance_type': ['gain']}
    
        
    xgb_clf=xgboost.XGBRFClassifier()
    grid_search=GridSearchCV(xgb_clf,param_grid=param_grid,cv=5,scoring='f1',verbose=2,n_jobs=-1,iid=True)
    grid_search.fit(X_train_trans,y_train)
    
    params=grid_search.best_params_
    
    #clf=xgboost.XGBRFClassifier(**params,n_estimators=100,n_jobs=-1,random_state=42)
    clf=xgboost.XGBRFClassifier(**params)
    clf.fit(X_train_trans,y_train)
    
    y_test=test['bowties']
    X_test=test.drop(columns='bowties')
    X_test=pipeline.fit_transform(X_test)
    
    y_preds=clf.predict(X_test)
    
    F_CV=grid_search.best_score_    
    P,R,F=precision_score(y_test,y_preds),recall_score(y_test,y_preds),f1_score(y_test,y_preds)

    print(P,R,F,F_CV,params)
    #0.8374384236453202 0.8808290155440415 0.8585858585858585 0.8345152519028066 {'_Booster': None, 'base_score': 0.5, 'colsample_bylevel': 1, 'colsample_bynode': 0.8, 'colsample_bytree': 1, 'gamma': 0, 'importance_type': 'gain', 'learning_rate': 0.05, 'max_delta_step': 0, 'max_depth': 7, 'min_child_weight': 5, 'n_estimators': 100, 'n_jobs': 1, 'nthread': None, 'objective': 'binary:logistic', 'random_state': 42, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'silent': None, 'subsample': 0.8, 'verbosity': 1}
    #0.8366336633663366 0.8756476683937824 0.8556962025316456 0.8352625965436676 {'_Booster': None, 'base_score': 0.5, 'colsample_bylevel': 1, 'colsample_bynode': 0.8, 'colsample_bytree': 1, 'gamma': 0, 'importance_type': 'gain', 'learning_rate': 0.05, 'max_delta_step': 0, 'max_depth': 6, 'min_child_weight': 5, 'n_estimators': 100, 'n_jobs': 1, 'nthread': None, 'objective': 'binary:logistic', 'random_state': 42, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'silent': None, 'subsample': 0.8, 'verbosity': 1}
    #0.8439024390243902 0.8963730569948186 0.8693467336683416 0.8497798540354795 {'_Booster': None, 'base_score': 0.5, 'colsample_bylevel': 1, 'colsample_bynode': 0.2, 'colsample_bytree': 1, 'gamma': 0, 'importance_type': 'gain', 'learning_rate': 0.05, 'max_delta_step': 0, 'max_depth': 7, 'min_child_weight': 5, 'n_estimators': 100, 'n_jobs': 1, 'nthread': None, 'objective': 'binary:logistic', 'random_state': 42, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'silent': None, 'subsample': 0.8, 'verbosity': 1}
    
final_params_selected=True
if final_params_selected:
    joblib.dump(clf,"C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\classifiers\\XGBRF_img_classifier.pkl")

    