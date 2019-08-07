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

import matplotlib.pyplot as plt

os.chdir('C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification')
import data_prep as dp

from imp import reload
reload(dp)



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

seeking=True

if seeking:
    max_pool_performance=dict()
    for pool_side_length in [1,2,4,8]:
        print('Starting pool_side_length='+str(pool_side_length))
        
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
        
        #Testing:
        #X_train_trans=dp.max_pooling(pool_side_length=2,image_side_length=8).transform(X_train)
        
        pipeline=Pipeline([('Max_Pooling',dp.max_pooling(pool_side_length=pool_side_length,image_side_length=8)),
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
        
        #clf=SVC(C=params['C'],gamma=params['gamma'],kernel='rbf')
        clf=SVC(**params)
        clf.fit(X_train_trans,y_train)
        
        y_test=test['bowties']
        X_test=test.drop(columns='bowties')
        X_test=pipeline.fit_transform(X_test)
        
        y_preds=clf.predict(X_test)
        
        F_CV=rand_search.best_score_    
        P,R,F=precision_score(y_test,y_preds),recall_score(y_test,y_preds),f1_score(y_test,y_preds)
        
        max_pool_performance[int(pool_side_length**2)]=(P,R,F,F_CV,params) #Precision, Recall, F1 (when applied to test set), F1_CV (measured during cross validation training across 150 trials, parameters used for both)
    
        print(str(pool_side_length),'| P:',str(round(P,2)),'R:',str(round(R,2)),'F:',str(round(F,2)),'F_CV:',str(round(F_CV,2)))
        
    joblib.dump(max_pool_performance,"C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\classifiers\\SVM_img_max-pool_dict.pkl")


if seeking==False:
    max_pool_performance=joblib.load(max_pool_performance,"C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\classifiers\\SVM_img_max-pool_dict.pkl")


################################################
# PLOT PERFORMANCE SCORE (F1) OF EACH SVC
# VERSUS THE RESOLUTION REDUCTION VALUE
################################################
#Using res_reduct_preformance dictionary calculated above
pool_size=[pool_size for pool_size in max_pool_performance]


prec_vals_test=[max_pool_performance[r][0] for r in max_pool_performance]
rec_values_test=[max_pool_performance[r][1] for r in max_pool_performance]
f1_values_test=[max_pool_performance[r][2] for r in max_pool_performance]
f1_values_CV=[max_pool_performance[r][3] for r in max_pool_performance]

plt.close('all')
plt.figure()
plt.plot(pool_size,prec_vals_test,'ro',alpha=0.7)
plt.plot(pool_size,rec_values_test,'bo',alpha=0.7)
plt.plot(pool_size,f1_values_test,'go',alpha=0.7)
plt.plot(pool_size,f1_values_CV,'ko',alpha=0.7)
plt.legend(['Precision_test','Recall_test','F1_test','F1_CV'])
plt.ylim(0.65,0.95)
plt.grid()
plt.xlabel('Number of Pixels in Max Pool')
plt.ylabel('SVC Score')

plt.figure(2)
plt.plot(pool_size,f1_values_test,'go',alpha=0.7)
plt.plot(pool_size,f1_values_CV,'ko',alpha=0.7)
plt.grid()
plt.xlabel('Number of Pixels in Max Pool')
plt.ylabel('SVC Score')
plt.legend(['F1_test','F1_CV'])



final_params_selected=True
pool_side_length=2 #corresponds to 18 features in each circle sweep
params=max_pool_performance[int(pool_side_length**2)][4] #{'C': 16.22388397086384, 'gamma': 0.20261142283225705, 'kernel': 'rbf'}

if final_params_selected:
    X=numpy_to_pd(X_raw,column_names)
    
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
    
    pipeline=Pipeline([('Max_Pooling',dp.max_pooling(pool_side_length=pool_side_length,image_side_length=8)),
                       ('Imputer',SimpleImputer(strategy='mean')),
                       ('Scaler',StandardScaler()),
                       ])
    
    
    X_train_trans=pipeline.fit_transform(X_train)
        
    clf=SVC(C=params['C'],gamma=params['gamma'],kernel='rbf')
    clf.fit(X_train_trans,y_train)
    
    joblib.dump(clf,"C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\classifiers\\SVM_img_max-pool-"+str(int(pool_side_length**2))+"_classifier.pkl")

    

