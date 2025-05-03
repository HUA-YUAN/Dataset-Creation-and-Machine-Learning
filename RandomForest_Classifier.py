# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 13:18:07 2025

@author: stanl
"""

# 外加模組
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import numpy as np
from statistics import mean 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


# 自訂函式
import Draw_RandomForest_Result



# 模型(交叉驗證中的訓練集與驗證集)
def classifier_cross_validation(seed, x_train_val, y_train_val, stratified, for_stratified_magnitude, mode, max_depth, n_estimators, feature):
   #Seed Number To make sure the productivity of the model
   np.random.seed(seed)
    
   Accuracy_magnitude_train = []
   Accuracy_magnitude_val = []  
   
   if mode == "gridsearch":
       rf_reg = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features='sqrt')
       
   if mode == "best_parameters":
       best_param_rf_reg = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features='sqrt')

   # 於原先的訓練集中劃分出訓練集與驗證集做交叉驗證
   for train_index, val_index in stratified.split(x_train_val, for_stratified_magnitude):
           
       #print("%s %s" % (train_index,test_index))
       x_train_fold, x_val_fold = x_train_val[train_index], x_train_val[val_index]
       y_train_fold, y_val_fold = y_train_val[train_index], y_train_val[val_index] 
       
       if mode == "gridsearch":
           # 模型的訓練與驗證
          predicted_value1, predicted_value2, rf_reg = classifier_taining(rf_reg, x_train_fold, y_train_fold, x_val_fold)

       if mode == "best_parameters":  
           # 模型的訓練與驗證
           predicted_value1, predicted_value2, best_param_rf_reg = classifier_taining(best_param_rf_reg, x_train_fold, y_train_fold, x_val_fold)
           
           # 特徵重要性分析
           randomforest_feature_importances(best_param_rf_reg, feature)
                     
           data_set = "train"
           cm_train = confusion_matrix(y_train_fold, predicted_value1)
           Draw_RandomForest_Result.draw_classifier_confusion_matrix(cm_train, data_set)
           
           data_set = "val"
           cm_val = confusion_matrix(y_val_fold, predicted_value2)
           Draw_RandomForest_Result.draw_classifier_confusion_matrix(cm_val, data_set)
           
       # 計算訓練集預測誤差
       accuracy_magnitude_train_prediction = accuracy_score(y_train_fold,predicted_value1)
       #print('Train Accuracy of Magnitude: %.4f' % (accuracy_magnitude_train_prediction))
       Accuracy_magnitude_train.append(accuracy_magnitude_train_prediction)
       accuracy_magnitude_val_prediction = accuracy_score(y_val_fold,predicted_value2)
       #print('Test Accuracy of Magnitude: %.4f' % (accuracy_magnitude_test_prediction))
       Accuracy_magnitude_val.append(accuracy_magnitude_val_prediction)
      
   if mode == "gridsearch": 
       train_score1 = mean(Accuracy_magnitude_train) 
       val_score1 = mean(Accuracy_magnitude_val)   
       return train_score1, val_score1 
       
   if mode == "best_parameters": 
       return best_param_rf_reg, Accuracy_magnitude_train, Accuracy_magnitude_val

   
   
def randomforest_feature_importances(best_param_rf_reg, feature):
    # 特徵重要性分析
    feature_importances = best_param_rf_reg.feature_importances_   
    if feature == "A":
        col_names = ["Max","Min","P2P","Mean(ABS)","STD"]
    if feature == "B":
        col_names = ["Pa","Pv","Pd","CAV"]
    if feature == "C":
        col_names = ["P2P","Mean(ABS)","STD","Pa","Pv","Pd","CAV"]
        
    plt.figure(dpi=150)
    importances_bar = plt.bar(col_names,feature_importances,color='indigo')
    plt.bar_label(importances_bar,fmt='%.4f',label_type='edge')
    plt.title('Feature Importances',fontsize=20, fontweight='bold')
    plt.ylim(0,0.5) 
    plt.show() 
    
    
    
def classifier_taining(rf_reg, x_train_fold, y_train_fold, x_predicted_fold):
    rf_reg.fit(x_train_fold, y_train_fold)
    #Train 與 Val 的預測結果
    predicted_value1 = np.round(rf_reg.predict(x_train_fold))
    predicted_value2 = np.round(rf_reg.predict(x_predicted_fold)) 
    # np.round()震度四捨五入 
    
    return predicted_value1, predicted_value2, rf_reg



# 模型(合併後的訓練集與測試集)
def classifier_testing(seed, max_depth, n_estimators, x_test, y_test, x_train_val, y_train_val, feature): 
    #Seed Number To make sure the productivity of the model
    np.random.seed(seed)
    best_param_rf_reg_final = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features='sqrt')
    
    # Train 與 Test 的預測結果    
    final_predicted_value1, predicted_value3, best_param_rf_reg_final = classifier_taining(best_param_rf_reg_final, x_train_val, y_train_val, x_test)
        
    # 特徵重要性分析
    randomforest_feature_importances(best_param_rf_reg_final, feature)
    
    data_set = "frinal_train"
    cm_final_train = confusion_matrix(y_train_val, final_predicted_value1)   
    Draw_RandomForest_Result.draw_classifier_confusion_matrix(cm_final_train, data_set)
    
    data_set = "test"
    cm_test = confusion_matrix(y_test, predicted_value3)   
    Draw_RandomForest_Result.draw_classifier_confusion_matrix(cm_test, data_set)
     
    Accuracy_magnitude_final_train = accuracy_score(y_train_val, final_predicted_value1)    
    Accuracy_magnitude_test = accuracy_score(y_test, predicted_value3)   
    
    return best_param_rf_reg_final, Accuracy_magnitude_final_train, Accuracy_magnitude_test



    