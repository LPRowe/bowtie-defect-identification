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
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

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
# DICTIONARY CONNECTING {CLASSIFER_NAME:[CLASSIFIER,TEST DATA],}
# Dictionary is updated later in script to include p and y
# Respectively probability of (non)bowtie and true classification (1|0)
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
def predict(clf,data):
    '''
    Takes the classifier and test data for each classifier created
    All of the test data is the same and in the same order
    except it has been transformed differently for each classifier
    

    returns:
        array p=[proba nonbowtie, proba bowtie] probability of nonbowtie or bowtie prediction
        array p_hard=[0,1] for bowtie and [1,0] for nonbowtie predictionand 
        1D array y=[true classification 1 for bowtie and 0 for nonbowtie]
    '''
    try:
        y=data['bowties']
        X=data.drop(columns='bowties')
    except:
        data=np.array(data)
        X,y=data[:,:-1],data[:,-1]
        

    try:
        p_hard=clf.predict(X)
        p_soft=clf.predict_proba(X)
    except:
        #For some reason bowtie value column is not always dropped this will force the drop
        X=np.array(data)[:,:-1]
        p_hard=clf.predict(X)
        p_soft=clf.predict_proba(X)
        
    return(p_soft,p_hard,y)

    

# =============================================================================
# Append predictions, true classification, F1 score to clf_test_data dictionary
# =============================================================================
for clf_ in clf_test_data:
    clf=clf_test_data[clf_][0]
    data=clf_test_data[clf_][1]
    #globals()['p_%s'%clf_],globals()['y_%s'%clf_]=predict(clf,data)
    p_soft,p_hard,y=predict(clf,data)
    
    # =========================================================================
    # Calculate F1 for each classifier to use for weighted voting
    # =========================================================================
    #F1=soft_eval_to_f1(p,y)
    F1=f1_score(p_hard,y)
          
    clf_test_data[clf_]=[clf,data,p_soft,p_hard,y,F1]
    
# =============================================================================
# Create our soft voter with predictions weighted by F1 score
# =============================================================================
def soft_vote(ctd=clf_test_data,weight='f1pow',power=20):
    #[0  ,1   ,2     ,3     ,4,5 ]
    #[clf,data,p_soft,p_hard,y,F1]
    
    #Sum all F1 values
    sum_weight=0
    for clf in ctd:
        if weight=='f1':
            sum_weight+=ctd[clf][5]
        elif weight=='uniform':
            sum_weight+=1
        elif weight=='f1pow':
            sum_weight+=ctd[clf][5]**power
        
    #For instance 1 p_pos[0]=[f1*p1[0]+f2*p2[0]+...fn*pn[0]] for each of the classifiers
    #For instance 2 p_pos[1]=[f1*p1[1]+f2*p2[1]...fn*pn[1]] for each of the classifiers
    #Where fn is the f1 value for the nth classifier and pn is the nth predicted probability
    p_pos=np.linspace(0,0,len(ctd['ET_img'][4]))
    p_neg=np.linspace(0,0,len(ctd['ET_img'][4]))
    
    for clf in ctd:
        count=0
        for p in np.array(ctd[clf][2]):
            if weight=='f1':
                w=np.array(ctd[clf][5])
            elif weight=='uniform':
                w=1
            elif weight=='f1pow':
                w=ctd[clf][5]**power
            p_pos[count]+=p[1]*w
            p_neg[count]+=p[0]*w
            count+=1
    
    p_pos=p_pos/sum_weight #divide all probabilities by the sum of all classifiers f1 scores
    p_neg=p_neg/sum_weight
    
    return(p_neg,p_pos)

p_neg,p_pos=soft_vote(clf_test_data,weight='f1pow',power=20)


# =============================================================================
# Predict bowtie or non-bowtie based on p_pos
# =============================================================================
def predict_y(p_pos,p_crit):
    y_pred=np.array(p_pos)
    y_pred[y_pred>=p_crit]=1
    y_pred[y_pred<p_crit]=0
    return y_pred.tolist()

y_true=[int(i) for i in clf_test_data['ET_img'][4].tolist()]
p_crit_vals=np.linspace(0.01,0.99,500)


p_soft=[]
r_soft=[]
f_soft=[]
for p_crit in p_crit_vals:
    p_soft.append(precision_score(predict_y(p_pos,p_crit),y_true))
    r_soft.append(recall_score(predict_y(p_pos,p_crit),y_true))
    f_soft.append(f1_score(predict_y(p_pos,p_crit),y_true))

save_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\side-scripts\\pr-curves'  
os.chdir(save_dir)
for clf_name in clf_test_data:
    print(clf_name)
    #[0  ,1   ,2     ,3     ,4,5 ]
    #[clf,data,p_soft,p_hard,y,F1]
    
    p_soft=clf_test_data[clf_name][2]
    p_pos=[i[1] for i in p_soft]
    
    precision=[]
    recall=[]
    f1=[]
    for p_crit in p_crit_vals:
        precision.append(precision_score(predict_y(p_pos,p_crit),y_true))
        recall.append(recall_score(predict_y(p_pos,p_crit),y_true))
        f1.append(f1_score(predict_y(p_pos,p_crit),y_true))
    
    plt.close('all')
    plt.figure(1)
    plt.plot(p_crit_vals,precision,'r-',lw=2)
    plt.plot(p_crit_vals,recall,'g-',lw=2)
    plt.plot(p_crit_vals,f1,'b-',lw=2)
    plt.xlabel('Bowtie Probability Cutoff')
    plt.legend(['Precision','Recall','F1 Score'])
    plt.title('P-R Scores for '+str(clf_name))
    plt.savefig(str(clf_name)+'_PR_scores.png')
    
    
    plt.figure(2)
    plt.plot(recall,precision,'b-',lw=2)
    plt.xlabel('Recall (False Positive Rate)')
    plt.ylabel('Precision (True Positive Rate)')
    plt.title('Precision-Recall for '+str(clf_name))
    plt.savefig(str(clf_name)+'_PR_curve.png')
    
    
    globals()['p_%s'%clf_name]=precision
    globals()['r_%s'%clf_name]=recall
    globals()['f_%s'%clf_name]=f1

plt.close('all')
plt.figure(1)
lw=2
plt.plot(p_crit_vals,r_XGBC_img,'b-',lw=lw)
plt.plot(p_crit_vals,r_XGBRFC_img,'g--',lw=lw)
plt.plot(p_crit_vals,p_XGBC_img,'c-',lw=lw)
plt.plot(p_crit_vals,p_XGBRFC_img,'r--',lw=lw)
plt.legend(['XGBC Recall','XGBRFC Recall','XGBC Precision','XGB-RFC Precision'])
plt.xlabel('Bowtie Probability Cutoff')
plt.savefig('XGBC-XGBRFC_precision_recall.png')

'''
plt.figure(2)
plt.plot(p_crit_vals,p_XGBC_img,'b-')
plt.plot(p_crit_vals,p_XGBRFC_img,'g--')
plt.legend(['XGBC Precision','XGB-RFC Precision'])
plt.xlabel('Bowtie Probability Cutoff')
plt.savefig('XGBC-XGBRFC_Precision.png')
'''

plt.figure(3)
plt.plot(p_crit_vals,f_XGBC_img,'b-')
plt.plot(p_crit_vals,f_XGBRFC_img,'g--')
plt.legend(['XGBC F1','XGB-RFC F1'])
plt.xlabel('Bowtie Probability Cutoff')
plt.savefig('XGBC-XGBRFC_F1.png')

plt.figure(4)
for clf_name in clf_test_data:
    plt.plot(p_crit_vals,globals()['f_%s'%clf_name])
plt.plot(p_crit_vals,f_soft)
legend=[clf_name for clf_name in clf_test_data]
legend.append('Soft Voter uniform')
plt.legend(legend)
plt.xlabel('Bowtie Probability Cutoff')
plt.ylabel('F1 Score')
plt.savefig('f1pow_score_all_classifiers.png')
