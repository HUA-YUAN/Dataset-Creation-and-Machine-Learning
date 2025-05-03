# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:24:20 2025

@author: stanl
"""


import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


# 用來判斷於訓練集中所需要生成虛擬資料之震度和數量的標準(重新分類原先訓練集資料的震度) 
def classify_earthquake_magnitude_smote(y_train_fold):
    y_train_magnitude_smote = []  

    for j in y_train_fold:
        if j >= 400:
            magnitude_j = 7
        #elif j >= 250 and j < 400:
         #   magnitude_j = 6
        #將5與6級分為同一類
        elif j >= 250 and j < 400:
            magnitude_j = 5
        elif j >= 80 and j < 250:
            magnitude_j = 5
        elif j >= 25 and j < 80:
            magnitude_j = 4
        elif j >= 8 and j < 25:
            magnitude_j = 3
        elif j >= 2.5 and j < 8:
            magnitude_j = 2     
        else:
            magnitude_j = 1                
        y_train_magnitude_smote.append(magnitude_j)
    
    return  y_train_magnitude_smote


def data_smote(x_train_fold,  y_train_fold):
    y_train_magnitude_smote = classify_earthquake_magnitude_smote(y_train_fold)
    y_train_magnitude_smote = LabelEncoder().fit_transform(y_train_magnitude_smote)
         
    y_train_fold = np.array(y_train_fold).reshape(len(y_train_fold),1)

    train_fold_do_smote = np.concatenate([x_train_fold, y_train_fold], axis=1)
    oversample = SMOTE()
    train_fold_smote, y_train_magnitude_smote_after = oversample.fit_resample(train_fold_do_smote, y_train_magnitude_smote)
         
    x_train_fold, y_train_fold = train_fold_smote[:, :-1], train_fold_smote[:, -1]
    
    return x_train_fold,  y_train_fold