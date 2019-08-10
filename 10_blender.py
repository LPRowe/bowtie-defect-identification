# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:53:10 2019

@author: Logan Rowe
"""


# =============================================================================
# LOAD DATA FOR CIRCLE SWEEP AND BOWTIE IMAGES
# =============================================================================

data_dir='C:\\Users\\Logan Rowe\\Desktop\\bowtie-defect-identification\\Wafer_Images\\bowtie-training-data'
X_raw_img=np.load(data_dir+'\\std0_std45_sh0-arr_sh45-arr_bow-bool_train.npy')
X_raw_cir=np.load(data_dir+'\\thetaM_theta0_theta45_std0_std45_sh0_sh45_bow-bool_train_144_dim.npy')

# =============================================================================
#  ADD COLUMN NAMES AND CONVERT TO PANDAS DF
# =============================================================================




#Circle Scan Column Names
c1=['thetaM','theta0','theta45','std0','std45']
c2=['sh0_{0}'.format(str(i)) for i in range(72)]
c3=['sh45_{0}'.format(str(i)) for i in range(72)]
c4=['bowties']
column_names=c1+c2+c3+c4