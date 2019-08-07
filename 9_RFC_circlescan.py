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


################################################
# LOAD DATA
################################################
data_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowtie-training-data'
X_raw=np.load(data_dir+'\\thetaM_theta0_theta45_std0_std45_sh0_sh45_bow-bool_train_144_dim.npy')


################################################
# ADD COLUMN NAMES AND CONVERT TO PANDAS DF
################################################

c1=['thetaM','theta0','theta45','std0','std45']
c2=['sh0_{0}'.format(str(i)) for i in range(72)]
c3=['sh45_{0}'.format(str(i)) for i in range(72)]
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
    
    pipeline=Pipeline([('ThetaDiff',dp.combine_theta_peaks(combine_and_remove=False)),
                       ('Imputer',SimpleImputer(strategy='mean')),
                       ])
    
    
    X_train_trans=pipeline.fit_transform(X_train)
    
    '''
    param_distribs={'criterion':['gini','entropy'],
                    'max_depth':[6,8,10,12,14],
                    }
    '''
    param_grid={'max_depth':[None],
                'bootstrap':[True,False],
                'criterion':['entropy','gini'],
                'min_samples_leaf':[5]
                }
    
    
        
    rnd_clf=RandomForestClassifier(n_estimators=50,n_jobs=-1,max_features='log2',random_state=42)
    grid_search=GridSearchCV(rnd_clf,param_grid=param_grid,cv=5,scoring='f1',verbose=2,n_jobs=-1,iid=True)
    grid_search.fit(X_train_trans,y_train)
    
    params=grid_search.best_params_
    
    clf=RandomForestClassifier(criterion=params['criterion'],bootstrap=params['bootstrap'],max_depth=params['max_depth'],n_estimators=500,n_jobs=-1,max_features='log2',random_state=42)
    clf.fit(X_train_trans,y_train)
    
    y_test=test['bowties']
    X_test=test.drop(columns='bowties')
    X_test=pipeline.fit_transform(X_test)
    
    y_preds=clf.predict(X_test)
    
    F_CV=grid_search.best_score_    
    P,R,F=precision_score(y_test,y_preds),recall_score(y_test,y_preds),f1_score(y_test,y_preds)

    print(P,R,F,F_CV,params)
    #0.7772277227722773 0.8134715025906736 0.7949367088607595 0.7961640752734754 {'bootstrap': True, 'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 5}
    

final_params_selected=True
if final_params_selected:
    joblib.dump(clf,"C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\classifiers\\RF_circle_sweep_classifier")

    