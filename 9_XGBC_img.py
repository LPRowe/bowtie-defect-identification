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
    X_val_trans=pipeline.fit_transform(X_val)
    X_test_trans=pipeline.fit_transform(X_test)
    
    # =========================================================================
    # Convert back to panda dataframe because XGBoost and Scipy dont play nice
    # =========================================================================
    
    column_names_xgb=['f{}'.format(int(i)) for i in range(130)]
    
    X_train_trans=dp.numpy_to_pd(X_train_trans,column_names_xgb)
    X_val_trans=dp.numpy_to_pd(X_val_trans,column_names_xgb)
    X_test_trans=dp.numpy_to_pd(X_test_trans,column_names_xgb)

    
    
    param_grid={ 'gamma':[0.05], 
                'learning_rate':[0.1],
                'max_depth':[7], 
                'min_child_weight':[1], 
                'n_estimators':[50], 
                'n_jobs':[-1], 
                'objective':['binary:logistic'], 
                'random_state':[42], 
                'reg_alpha':[0],
                'reg_lambda':[1], 
                'scale_pos_weight':[1], 
                'subsample':[1], 
                'verbosity':[1]
                }
    
        
    xgb_clf=xgboost.XGBClassifier()
    grid_search=GridSearchCV(xgb_clf,param_grid=param_grid,cv=5,scoring='f1',verbose=2,n_jobs=-1,iid=True)
    grid_search.fit(X_train_trans,y_train)
    
    params=grid_search.best_params_
    
    #clf=xgboost.XGBRFClassifier(**params,n_estimators=100,n_jobs=-1,random_state=42)
    clf=xgboost.XGBClassifier(**params)
    clf.fit(X_train_trans,y_train,early_stopping_rounds=10,eval_set=[(X_val_trans,y_val)])
    
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
    
# =============================================================================
#  Run again with more estimators and early stopping to check for over fitting
# =============================================================================
params['n_estimators']=100

clf=xgboost.XGBClassifier(**params)
eval_set=[(X_train_trans,y_train),(X_val_trans,y_val),(X_test_trans,y_test)]
eval_metric=['error','logloss']
clf.fit(X_train_trans,y_train,eval_metric=eval_metric,eval_set=eval_set,verbose=10)

evals_result=clf.evals_result()

#Errors
train_errors=evals_result['validation_0']['error']
val_errors=evals_result['validation_1']['error']
test_errors=evals_result['validation_2']['error']

#Logloss Errors
train_errors_log=evals_result['validation_0']['logloss']
val_errors_log=evals_result['validation_1']['logloss']
test_errors_log=evals_result['validation_2']['logloss']

N=np.linspace(1,params['n_estimators'],params['n_estimators'])

plt.close('all')
#Plot error 
plt.figure(1)
plt.plot(N,val_errors,'b-')
plt.plot(N,train_errors,'r-')
plt.plot(N,test_errors,'g-')
plt.legend(['Validation','Training','Testing'])
plt.xlabel('Number of Estimators')
plt.ylabel('Error')

#Plot logloss error
plt.figure(2)
plt.plot(N,val_errors_log,'b-')
plt.plot(N,train_errors_log,'r-')
plt.plot(N,test_errors_log,'g-')
plt.legend(['Validation','Training','Testing'])
plt.xlabel('Number of Estimators')
plt.ylabel('Logloss Error')

y_preds=clf.predict(X_test)

F_CV=grid_search.best_score_    
P,R,F=precision_score(y_test,y_preds),recall_score(y_test,y_preds),f1_score(y_test,y_preds)

print(P,R,F,F_CV,params)

# =============================================================================
#  Based on the logloss curves the optimal number of estimators is
#  between 46 and 58 so we will run it for 70 and use early stopping
# =============================================================================
params['n_estimators']=70
final_params_selected=True
if final_params_selected:
    # =============================================================================
    #  Combine training and validation sets to increase training data
    # =============================================================================
    X_train_full=pd.concat([X_train_trans,X_val_trans])
    y_train_full=pd.concat([y_train,y_val])
    
    clf=xgboost.XGBClassifier(**params)
    eval_set=[(X_train_trans,y_train),(X_test_trans,y_test)]
    eval_metric=['error','logloss']
    clf.fit(X_train_full,y_train_full,eval_metric=eval_metric,eval_set=eval_set,verbose=5,early_stopping_rounds=5)
    
    evals_result=clf.evals_result()
    
    #Logloss Errors
    train_errors_log_2=evals_result['validation_0']['logloss']
    test_errors_log_2=evals_result['validation_1']['logloss']
    
    N_2=np.linspace(1,len(test_errors_log_2),len(test_errors_log_2))
    #Plot logloss error
    plt.figure(3)
    plt.plot(N_2,train_errors_log_2,'r-')
    plt.plot(N_2,test_errors_log_2,'g-')
    plt.legend(['Training','Testing'])
    plt.xlabel('Number of Estimators')
    plt.ylabel('Logloss Error')
    
    y_preds=clf.predict(X_test)
    
    F_CV=grid_search.best_score_    
    P,R,F=precision_score(y_test,y_preds),recall_score(y_test,y_preds),f1_score(y_test,y_preds)
    
    print(P,R,F,F_CV,params)
    
    
    joblib.dump(clf,"C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\classifiers\\XGBC_img_classifier.pkl")


export_full_transformed_dataset=True
if export_full_transformed_dataset:
    processed_data_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\preprocessed_datasets'
    
    #Training Data Set
    training_full=X_train_full
    training_full['bowties']=y_train_full
    joblib.dump(training_full,processed_data_dir+'\\XGBC_img_train.pkl')
    
    #Testing Data Set
    testing_full=test
    joblib.dump(testing_full,processed_data_dir+'\\XGBC_img_test.pkl')

    