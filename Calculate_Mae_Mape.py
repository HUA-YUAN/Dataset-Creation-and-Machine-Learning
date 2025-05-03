# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 17:19:18 2024

@author: stanl
"""

# 外加模組
import numpy as np


# 計算模型的預測誤差(交叉驗證中的訓練集與驗證集)
# 計算模型的預測誤差(最終訓練集與測試集) 
def mape_mae(y_fold, predicted, y_magnitude, data_set): 
    y_fold_1 = []
    y_fold_2 = []
    y_fold_3 = []
    y_fold_4 = []
    y_fold_5 = []
    y_fold_6 = []
    y_fold_7 = []
    
    predicted_1 = []
    predicted_2 = []
    predicted_3 = []
    predicted_4 = []
    predicted_5 = []
    predicted_6 = []
    predicted_7 = []  
    
    for k in range(len(y_magnitude)):
        if y_magnitude[k] == 7:
            y_fold_7.append(y_fold[k])
            predicted_7.append(predicted[k])
            
        elif y_magnitude[k] == 6:
            y_fold_6.append(y_fold[k])
            predicted_6.append(predicted[k])
            
        elif y_magnitude[k] == 5:
            y_fold_5.append(y_fold[k])
            predicted_5.append(predicted[k])
            
        elif y_magnitude[k] == 4:
            y_fold_4.append(y_fold[k])
            predicted_4.append(predicted[k])
            
        elif y_magnitude[k] == 3:
            y_fold_3.append(y_fold[k])
            predicted_3.append(predicted[k])
            
        elif y_magnitude[k] == 2:
            y_fold_2.append(y_fold[k])
            predicted_2.append(predicted[k]) 
            
        else:
            y_fold_1.append(y_fold[k])
            predicted_1.append(predicted[k])   
                           
    y_fold_1, predicted_1 = np.array(y_fold_1), np.array(predicted_1)
    mape_1 = np.mean(np.abs((y_fold_1 - predicted_1) / y_fold_1)) * 100
    mae_1 = np.sum(np.abs(y_fold_1 - predicted_1)) / len(y_fold_1)
    
    y_fold_2, predicted_2 = np.array(y_fold_2), np.array(predicted_2)
    mape_2 = np.mean(np.abs((y_fold_2 - predicted_2) / y_fold_2)) * 100
    mae_2 = np.sum(np.abs(y_fold_2 - predicted_2)) / len(y_fold_2)
    
    y_fold_3, predicted_3 = np.array(y_fold_3), np.array(predicted_3)
    mape_3 = np.mean(np.abs((y_fold_3 - predicted_3) / y_fold_3)) * 100
    mae_3 = np.sum(np.abs(y_fold_3 - predicted_3)) / len(y_fold_3)
    
    y_fold_4, predicted_4 = np.array(y_fold_4), np.array(predicted_4)
    mape_4 = np.mean(np.abs((y_fold_4 - predicted_4) / y_fold_4)) * 100
    mae_4 = np.sum(np.abs(y_fold_4 - predicted_4)) / len(y_fold_4)
    
    y_fold_5, predicted_5 = np.array(y_fold_5), np.array(predicted_5)
    mape_5 = np.mean(np.abs((y_fold_5 - predicted_5) / y_fold_5)) * 100
    mae_5 = np.sum(np.abs(y_fold_5 - predicted_5)) / len(y_fold_5)
    
    y_fold_6, predicted_6 = np.array(y_fold_6), np.array(predicted_6)
    mape_6 = np.mean(np.abs((y_fold_6 - predicted_6) / y_fold_6)) * 100
    mae_6 = np.sum(np.abs(y_fold_6 - predicted_6)) / len(y_fold_6)
    
    y_fold_7, predicted_7 = np.array(y_fold_7), np.array(predicted_7)
    mape_7 = np.mean(np.abs((y_fold_7 - predicted_7) / y_fold_7)) * 100
    mae_7 = np.sum(np.abs(y_fold_7 - predicted_7)) / len(y_fold_7)
    
    return  mape_1, mape_2, mape_3, mape_4, mape_5, mape_6, mape_7, \
            mae_1, mae_2, mae_3, mae_4, mae_5, mae_6, mae_7
            

        

      