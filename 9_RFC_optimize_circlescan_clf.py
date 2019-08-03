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

#if seeking optimal parameters for resolution reduction then True
seeking=True

if seeking:
    res_reduct_performance=dict()
    for max_leaf_nodes in [6,8,10,12,16,20,30,40,60,100]:
        print('Starting resolution reduction of',str(res))
        
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
        
        pipeline=Pipeline([('Reducer',dp.reduce_features_in_sweep(first_index_of_sweep_in_X=list(X.columns).index('sh0_0'),
                                      reduced_circle_sweep_res=res)),
                           ('ThetaDiff',dp.combine_theta_peaks(combine_and_remove=False)),
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

    

################################################
# PLOT PERFORMANCE SCORE (F1) OF EACH SVC
# VERSUS THE RESOLUTION REDUCTION VALUE
################################################
try:
    #Using res_reduct_preformance dictionary calculated above
    res_reduct=[r for r in res_reduct_performance]
except:
    #Loading an already saved version of the res_reduct_performance dictionary
    res_reduct_performance=joblib.load("C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\classifiers\\SVM_circlesweep_res_dict.pkl")
    res_reduct=[r for r in res_reduct_performance]

prec_vals_test=[res_reduct_performance[r][0] for r in res_reduct_performance]
rec_values_test=[res_reduct_performance[r][1] for r in res_reduct_performance]
f1_values_test=[res_reduct_performance[r][2] for r in res_reduct_performance]
f1_values_CV=[res_reduct_performance[r][3] for r in res_reduct_performance]

n_features=[72/r for r in res_reduct]

plt.close('all')
plt.figure()
plt.plot(n_features,prec_vals_test,'ro',alpha=0.7)
plt.plot(n_features,rec_values_test,'bo',alpha=0.7)
plt.plot(n_features,f1_values_test,'go',alpha=0.7)
plt.plot(n_features,f1_values_CV,'ko',alpha=0.7)
plt.legend(['Precision_test','Recall_test','F1_test','F1_CV'])
plt.ylim(0.65,0.95)
plt.grid()
plt.xlabel('Number of Features in Circle Sweep')
plt.ylabel('SVC Score')

plt.figure(2)
plt.plot(n_features,f1_values_test,'go',alpha=0.7)
plt.plot(n_features,f1_values_CV,'ko',alpha=0.7)
plt.grid()
plt.xlabel('Number of Features in Each Circle Sweep')
plt.ylabel('SVC Score')
plt.legend(['F1_test','F1_CV'])

    

final_params_selected=True
res=4 #corresponds to 18 features in each circle sweep
params=res_reduct_performance[res][4] #{'C': 16.22388397086384, 'gamma': 0.20261142283225705, 'kernel': 'rbf'}

if final_params_selected:
    X=dp.numpy_to_pd(X_raw,column_names)
    
    ################################################
    # SPLIT DATA INTO TEST AND TRAIN BOTH BALANCED
    # WITH RESPECT TO (NON)BOWTIES
    # RETAINING SPLIT FOR ENSEMBLE PURPOSES
    ################################################
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index, test_index in split.split(X,X['bowties']):
        train=X.loc[train_index]
        test=X.loc[test_index]
    
    y_train=train['bowties']  
    X_train=train.drop(columns='bowties')
    
    pipeline=Pipeline([('Reducer',dp.reduce_features_in_sweep(first_index_of_sweep_in_X=list(X.columns).index('sh0_0'),
                                  reduced_circle_sweep_res=res)),
                       ('ThetaDiff',dp.combine_theta_peaks(combine_and_remove=False)),
                       ('Imputer',SimpleImputer(strategy='mean')),
                       ('Scaler',StandardScaler()),
                       ])
    
    
    X_train_trans=pipeline.fit_transform(X_train)
        
    clf=SVC(C=params['C'],gamma=params['gamma'],kernel='rbf')
    clf.fit(X_train_trans,y_train)
    
    joblib.dump(clf,"C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\classifiers\\SVM_circlesweep_res-"+str(res)+"_classifier.pkl")

    