# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:06:06 2025

@author: stanl
"""

# 外加模組
import numpy as np


# 用來劃分交叉驗證中訓練集與驗證集資料的標準(重新分類原先訓練集資料的震度) 
def classify_magnitude_for_dataset(data):
    dataset_magnitude = []  

    for value in data:
        magnitude = earthquake_magnitude_standard(value)
        dataset_magnitude.append(magnitude)
        
    dataset_magnitude = np.array(dataset_magnitude)
       
    return dataset_magnitude  


# 分類交叉驗證中訓練集與驗證集資料的震度 
def classify_magnitude(y_fold, predicted, data_set):
    y_magnitude = []
    predicted_magnitude = []
    
    for value in y_fold:
        magnitude = earthquake_magnitude_standard(value)
        y_magnitude.append(magnitude)
        
    for value in predicted:
        magnitude = earthquake_magnitude_standard(value)
        predicted_magnitude.append(magnitude)  
        
    y_magnitude = np.array(y_magnitude)
    predicted_magnitude = np.array(predicted_magnitude)
    
    return y_magnitude, predicted_magnitude

    
def earthquake_magnitude_standard(value): 
    if value >= 400:
        magnitude = 7
    elif value >= 250 and value < 400:
        magnitude = 6
    elif value >= 80 and value < 250:
        magnitude = 5
    elif value >= 25 and value < 80:
        magnitude = 4
    elif value >= 8 and value < 25:
        magnitude = 3
    elif value >= 2.5 and value < 8:
        magnitude = 2     
    else:
        magnitude = 1 
        
    return magnitude
                  

    
    