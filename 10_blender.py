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
# DICTIONARY CONNECTING CLASSIFER TO TEST DATA
# =============================================================================
clf_data={ET_img:ETC_img_test,
          ETC_circlesweep:ETC_circle_test,
          RF_circle:RFC_circle_test,
          RF_img:RFC_img_test,
          SVM_circlesweep:SVC_circle_test,
          SVM_img:SVC_mp-img_test,
          XGBC_img:XGBC_img_test,
          XGBRFC_img:XGB-RFC_img_test,
        }