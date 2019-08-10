# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:53:10 2019

@author: Logan Rowe
"""

import os
import sys
import numpy as np
import pandas as pd
import glob

from sklearn.externals import joblib

# =============================================================================
# LOAD PREPROCESSED TEST DATA FOR EACH CLASSIFIER
# =============================================================================

data_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\preprocessed_datasets'
os.chdir(data_dir)
for file in glob.glob('*test.pkl'):
    globals()['%s'%file.split('.')[0]]=joblib.load(file)

# =============================================================================
# LOAD EACH CLASSIFIER
# =============================================================================

clf_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\classifiers'
os.chdir(clf_dir)
for file in glob.glob('*classifier.pkl'):
    globals()['%s'%'_'.join(file.split('_')[:2])]=joblib.load(file)
    
# =============================================================================
# DICTIONARY CONNECTING {CLASSIFER:TEST DATA,}
# =============================================================================
clf_test_data={'ET_img':[ET_img,ETC_img_test],
          'ETC_circlesweep':[ETC_circlesweep,ETC_circle_test],
          'RF_circle':[RF_circle,RFC_circle_test],
          'RF_img':[RF_img,RFC_img_test],
          'SVM_circlesweep':[SVM_circlesweep,SVC_circle_test],
          'SVM_img':[SVM_img,SVC_mpimg_test],
          'XGBC_img':[XGBC_img,XGBC_img_test],
          'XGBRFC_img':[XGBRFC_img,XGBRFC_img_test],
        }

# =============================================================================
# SPLIT DATA AND MAKE PREDICTIONS WITH CLASSIFIER
# =============================================================================
'''
def predict(clf,soft_voting=True):
    try:
        y=clf_test_data[clf][1]['bowties']
        X=clf_test_data[clf][1].drop(columns='bowties')
    except:
        data=np.array(clf_test_data[clf][1])
        X,y=data[:,:-1],data[:,-1]
        
    if soft_voting:
        p=clf.predict_proba(X)
    else:
        p=clf.predict(X)
    return(p,y)
'''
def predict(clf,data,soft_voting=True):
    try:
        y=data['bowties']
        X=data.drop(columns='bowties')
    except:
        data=np.array(data)
        X,y=data[:,:-1],data[:,-1]
        
    if soft_voting:
        p=clf.predict_proba(X)
    else:
        p=clf.predict(X)
    return(p,y)

for clf_ in clf_test_data:
    clf=clf_test_data[clf_][0]
    data=clf_test_data[clf_][1]
    globals()['p_%s'%clf_],globals()['y_%s'%clf_]=predict(clf,data)
    clf_test_data[clf_]=[clf,data,globals()['p_%s'%clf_],globals()['y_%s'%clf_]]

