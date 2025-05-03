# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 13:18:07 2025

@author: stanl
"""

# 外加模組
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean 
from sklearn.ensemble import RandomForestRegressor

# 自訂函式
import Calculate_Mae_Mape
import Classify_Earthquake_Magnitude
import Data_Smote
import Draw_RandomForest_Result


# 模型(交叉驗證中的訓練集與驗證集)
def regressor_cross_validation(seed, x_train_val, y_train_val, stratified, for_stratified_magnitude, mode, max_depth, n_estimators, feature, output_mode, smote_mode):
   #Seed Number To make sure the productivity of the model
   np.random.seed(seed)
    
   TrainScore_MAPE = []
   TrainScore_MAE = []
   ValScore_MAPE = []
   ValScore_MAE = []
   
   if mode == "gridsearch":
       rf_reg = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, max_features='sqrt')
       
   if mode == "best_parameters":
       TrainScore_MAPE_1 = []
       TrainScore_MAPE_2 = []
       TrainScore_MAPE_3 = []
       TrainScore_MAPE_4 = []
       TrainScore_MAPE_5 = []
       TrainScore_MAPE_6 = []
       TrainScore_MAPE_7 = []
       
       TrainScore_MAE_1 = []
       TrainScore_MAE_2 = []
       TrainScore_MAE_3 = []
       TrainScore_MAE_4 = []
       TrainScore_MAE_5 = []
       TrainScore_MAE_6 = []
       TrainScore_MAE_7 = []
       
       ValScore_MAPE_1 = []
       ValScore_MAPE_2 = []
       ValScore_MAPE_3 = []
       ValScore_MAPE_4 = []
       ValScore_MAPE_5 = []
       ValScore_MAPE_6 = []
       ValScore_MAPE_7 = []
       
       ValScore_MAE_1 = []
       ValScore_MAE_2 = []
       ValScore_MAE_3 = []
       ValScore_MAE_4 = []
       ValScore_MAE_5 = []
       ValScore_MAE_6 = []
       ValScore_MAE_7 = []
       
       Accuracy_magnitude_train = []
       Accuracy_magnitude_val = []
       
       best_param_rf_reg = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, max_features='sqrt')

   # 於原先的訓練集中劃分出訓練集與驗證集做交叉驗證
   for train_index, val_index in stratified.split(x_train_val, for_stratified_magnitude):
           
       #print("%s %s" % (train_index,test_index))
       x_train_fold, x_val_fold = x_train_val[train_index], x_train_val[val_index]
       y_train_fold, y_val_fold = y_train_val[train_index], y_train_val[val_index] 
       
       if smote_mode == "yes":
           # 內部生成虛擬資料     
           x_train_fold, y_train_fold = Data_Smote.data_smote(x_train_fold, y_train_fold)

       if mode == "gridsearch":
           # 模型的訓練與驗證
          predicted_value1, predicted_value2, rf_reg = regressor_taining(rf_reg, x_train_fold, y_train_fold, x_val_fold, output_mode)

       if mode == "best_parameters":  
           # 模型的訓練與驗證
           predicted_value1, predicted_value2, best_param_rf_reg = regressor_taining(best_param_rf_reg, x_train_fold, y_train_fold, x_val_fold, output_mode)
           
           # 特徵重要性分析
           randomforest_feature_importances(best_param_rf_reg, feature)
                     
           train_class, train_class0, train_class1, predicted_value1_class0, predicted_value1_class1 = Draw_RandomForest_Result.check_train_class(train_index, y_train_fold, predicted_value1)
           
           # 計算預測結果的震度
           data_set = "train"
           y_train_magnitude, predicted_value1_magnitude = Classify_Earthquake_Magnitude.classify_magnitude(y_train_fold, predicted_value1, data_set)
           
           train_mape_1, train_mape_2, train_mape_3, train_mape_4, train_mape_5, train_mape_6, train_mape_7, \
           train_mae_1, train_mae_2, train_mae_3, train_mae_4, train_mae_5, train_mae_6, train_mae_7 \
               = Calculate_Mae_Mape.mape_mae(y_train_fold, predicted_value1, y_train_magnitude, data_set)

           Draw_RandomForest_Result.draw_regressor_result_train(y_train_fold, predicted_value1, train_class0, train_class1, 
                                   predicted_value1_class0, predicted_value1_class1, data_set)
               
           Draw_RandomForest_Result.draw_regressor_magnitude_error(y_train_magnitude, predicted_value1_magnitude, data_set)
           
           data_set = "val"
           y_val_magnitude, predicted_value2_magnitude = Classify_Earthquake_Magnitude.classify_magnitude(y_val_fold, predicted_value2, data_set)
           
           val_mape_1, val_mape_2, val_mape_3, val_mape_4, val_mape_5, val_mape_6, val_mape_7, \
           val_mae_1, val_mae_2, val_mae_3, val_mae_4, val_mae_5, val_mae_6, val_mae_7 \
               = Calculate_Mae_Mape.mape_mae(y_val_fold, predicted_value2, y_val_magnitude, data_set)

           Draw_RandomForest_Result.draw_regressor_result_val_test(y_val_fold, predicted_value2, data_set)
           
           Draw_RandomForest_Result.draw_regressor_magnitude_error(y_val_magnitude, predicted_value2_magnitude, data_set)
               
            # "\"此符號為換行符號 
            
           TrainScore_MAPE_1.append(train_mape_1)
           TrainScore_MAPE_2.append(train_mape_2)
           TrainScore_MAPE_3.append(train_mape_3)
           TrainScore_MAPE_4.append(train_mape_4)
           TrainScore_MAPE_5.append(train_mape_5)
           TrainScore_MAPE_6.append(train_mape_6)
           TrainScore_MAPE_7.append(train_mape_7)
            
           TrainScore_MAE_1.append(train_mae_1)
           TrainScore_MAE_2.append(train_mae_2)
           TrainScore_MAE_3.append(train_mae_3)
           TrainScore_MAE_4.append(train_mae_4)
           TrainScore_MAE_5.append(train_mae_5)
           TrainScore_MAE_6.append(train_mae_6)
           TrainScore_MAE_7.append(train_mae_7)
            
           ValScore_MAPE_1.append(val_mape_1)
           ValScore_MAPE_2.append(val_mape_2)
           ValScore_MAPE_3.append(val_mape_3)
           ValScore_MAPE_4.append(val_mape_4)
           ValScore_MAPE_5.append(val_mape_5)
           ValScore_MAPE_6.append(val_mape_6)
           ValScore_MAPE_7.append(val_mape_7)
            
           ValScore_MAE_1.append(val_mae_1)
           ValScore_MAE_2.append(val_mae_2)
           ValScore_MAE_3.append(val_mae_3)
           ValScore_MAE_4.append(val_mae_4)
           ValScore_MAE_5.append(val_mae_5)
           ValScore_MAE_6.append(val_mae_6)
           ValScore_MAE_7.append(val_mae_7)
                             
           # 計算震度預測正確率
           accuracy_magnitude_train_prediction = np.count_nonzero(y_train_magnitude==predicted_value1_magnitude)/len(y_train_magnitude)
           Accuracy_magnitude_train.append(accuracy_magnitude_train_prediction)
           accuracy_magnitude_val_prediction = np.count_nonzero(y_val_magnitude==predicted_value2_magnitude)/len(y_val_magnitude)
           Accuracy_magnitude_val.append(accuracy_magnitude_val_prediction)
                       
       # 計算訓練集預測誤差
       y_train_fold, predicted_value1 = np.array(y_train_fold), np.array(predicted_value1)
       train_mape = np.mean(np.abs((y_train_fold - predicted_value1) / y_train_fold)) * 100
       train_mae = np.sum(np.abs(y_train_fold - predicted_value1)) / len(y_train_fold)
       TrainScore_MAPE.append(train_mape)
       TrainScore_MAE.append(train_mae)
       
       # 計算驗證集預測誤差
       y_val_fold, predicted_value2 = np.array(y_val_fold), np.array(predicted_value2)
       val_mape = np.mean(np.abs((y_val_fold - predicted_value2) / y_val_fold)) * 100
       val_mae = np.sum(np.abs(y_val_fold - predicted_value2)) / len(y_val_fold)
       ValScore_MAPE.append(val_mape)
       ValScore_MAE.append(val_mae)
   
      
   if mode == "gridsearch": 
       val_score1 = mean(ValScore_MAPE) 
       val_score2 = mean(ValScore_MAE)        
       train_score1 = mean(TrainScore_MAPE)
       train_score2 = mean(TrainScore_MAE)

       return val_score1, val_score2, train_score1, train_score2    
       
   if mode == "best_parameters":
       Model_Train_Mape_Result = {'TrainScore_MAPE': TrainScore_MAPE, 'TrainScore_MAPE_1': TrainScore_MAPE_1,
                                  'TrainScore_MAPE_2': TrainScore_MAPE_2, 'TrainScore_MAPE_3': TrainScore_MAPE_3, 
                                  'TrainScore_MAPE_4': TrainScore_MAPE_4, 'TrainScore_MAPE_5': TrainScore_MAPE_5,
                                  'TrainScore_MAPE_6': TrainScore_MAPE_6, 'TrainScore_MAPE_7': TrainScore_MAPE_7}
        
       Model_Validation_Mape_Result = {'ValScore_MAPE': ValScore_MAPE, 'ValScore_MAPE_1': ValScore_MAPE_1,
                                       'ValScore_MAPE_2': ValScore_MAPE_2, 'ValScore_MAPE_3': ValScore_MAPE_3,
                                       'ValScore_MAPE_4': ValScore_MAPE_4, 'ValScore_MAPE_5': ValScore_MAPE_5,
                                       'ValScore_MAPE_6': ValScore_MAPE_6, 'ValScore_MAPE_7': ValScore_MAPE_7}
    
       Model_Train_Mae_Result = {'TrainScore_MAE': TrainScore_MAE, 'TrainScore_MAE_1': TrainScore_MAE_1,
                                 'TrainScore_MAE_2': TrainScore_MAE_2, 'TrainScore_MAE_3': TrainScore_MAE_3,
                                 'TrainScore_MAE_4': TrainScore_MAE_4, 'TrainScore_MAE_5': TrainScore_MAE_5,
                                 'TrainScore_MAE_6': TrainScore_MAE_6, 'TrainScore_MAE_7': TrainScore_MAE_7}
        
       Model_Validation_Mae_Result = {'ValScore_MAE': ValScore_MAE, 'ValScore_MAE_1': ValScore_MAE_1,
                                      'ValScore_MAE_2': ValScore_MAE_2, 'ValScore_MAE_3': ValScore_MAE_3,
                                      'ValScore_MAE_4': ValScore_MAE_4, 'ValScore_MAE_5': ValScore_MAE_5,
                                      'ValScore_MAE_6': ValScore_MAE_6, 'ValScore_MAE_7': ValScore_MAE_7}
           
       Accuracy_Magnitude = {'Accuracy_magnitude_train': Accuracy_magnitude_train, 'Accuracy_magnitude_val': Accuracy_magnitude_val}

       return best_param_rf_reg, Model_Train_Mape_Result, Model_Train_Mae_Result, Model_Validation_Mape_Result, Model_Validation_Mae_Result, Accuracy_Magnitude
   
   
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
    
    
def regressor_taining(rf_reg, x_train_fold, y_train_fold, x_predicted_fold, output_mode):
    if output_mode == "pga":
        rf_reg.fit(x_train_fold, y_train_fold)
        #Train 與 Val 的預測結果
        predicted_value1 = np.round(rf_reg.predict(x_train_fold))
        predicted_value2 = np.round(rf_reg.predict(x_predicted_fold))

    if output_mode == "pga^2":
        rf_reg.fit(x_train_fold, np.power(y_train_fold,2))
        predicted_value1 = np.round(np.power(rf_reg.predict(x_train_fold),1/2))
        predicted_value2 = np.round(np.power(rf_reg.predict(x_predicted_fold),1/2))
        
    if output_mode == "log(pga)":
        rf_reg.fit(x_train_fold, np.log10(y_train_fold))
        predicted_value1 = np.round(np.power(10,rf_reg.predict(x_train_fold)))
        predicted_value2 = np.round(np.power(10,rf_reg.predict(x_predicted_fold)))   
        # np.round()震度四捨五入 
    
    return predicted_value1, predicted_value2, rf_reg


# 模型(合併後的訓練集與測試集)
def regressor_testing(seed, max_depth, n_estimators, x_test, y_test, x_train_val, y_train_val, output_mode, smote_mode, feature): 
    #Seed Number To make sure the productivity of the model
    np.random.seed(seed)
    best_param_rf_reg_final = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, max_features='sqrt')
    
    final_y_train_before_smote = y_train_val
    
    if smote_mode == "yes":
        # 內部生成虛擬資料
        x_train_val, y_train_val = Data_Smote.data_smote(x_train_val, y_train_val)
    
    # Train 與 Test 的預測結果    
    final_predicted_value1, predicted_value3, best_param_rf_reg_final = regressor_taining(best_param_rf_reg_final, x_train_val, y_train_val, x_test, output_mode)
        
    # 特徵重要性分析
    randomforest_feature_importances(best_param_rf_reg_final, feature)
    
    # 分區真實樣本與虛擬樣本(訓練集)   
    final_train_class, final_train_class0, final_train_class1, final_predicted_value1_class0, final_predicted_value1_class1 \
        = Draw_RandomForest_Result.check_train_class(final_y_train_before_smote, y_train_val, final_predicted_value1)

    # 計算預測誤差(訓練集)
    y_train_val, final_predicted_value1 = np.array(y_train_val), np.array(final_predicted_value1)
    final_train_mape = np.mean(np.abs((y_train_val - final_predicted_value1) / y_train_val)) * 100
    final_train_mae = np.sum(np.abs(y_train_val - final_predicted_value1)) / len(y_train_val)
    
    # 計算預測誤差(測試集)
    y_test, predicted_value3 = np.array(y_test), np.array(predicted_value3)
    test_mape = np.mean(np.abs((y_test - predicted_value3) / y_test)) * 100
    test_mae = np.sum(np.abs(y_test - predicted_value3)) / len(y_test)
    #test_error_STD = np.std(np.abs(y_test - predicted_value3))
    
    # 計算預測結果的震度
    data_set = "final_train"
    final_y_train_magnitude, final_predicted_value1_magnitude = Classify_Earthquake_Magnitude.classify_magnitude(y_train_val, final_predicted_value1, data_set)
    
    # 計算預測加速度值於各別震度的誤差(訓練集)
    final_train_mape_1, final_train_mape_2, final_train_mape_3, final_train_mape_4, final_train_mape_5, final_train_mape_6, final_train_mape_7, \
    final_train_mae_1, final_train_mae_2, final_train_mae_3, final_train_mae_4, final_train_mae_5, final_train_mae_6, final_train_mae_7 \
        = Calculate_Mae_Mape.mape_mae(y_train_val, final_predicted_value1, final_y_train_magnitude, data_set)
  
    Draw_RandomForest_Result.draw_regressor_result_train(y_train_val, final_predicted_value1, final_train_class0, final_train_class1,\
                        final_predicted_value1_class0, final_predicted_value1_class1, data_set)  

    Draw_RandomForest_Result.draw_regressor_magnitude_error(final_y_train_magnitude, final_predicted_value1_magnitude, data_set)  
    
    
    data_set = "test"
    y_test_magnitude, predicted_value3_magnitude = Classify_Earthquake_Magnitude.classify_magnitude(y_test, predicted_value3, data_set)
     
    # 計算預測加速度值於各別震度的誤差(測試集)
    test_mape_1, test_mape_2, test_mape_3, test_mape_4, test_mape_5, test_mape_6, test_mape_7, \
    test_mae_1, test_mae_2, test_mae_3, test_mae_4, test_mae_5, test_mae_6, test_mae_7 \
        = Calculate_Mae_Mape.mape_mae(y_test, predicted_value3, y_test_magnitude, data_set)

    Draw_RandomForest_Result.draw_regressor_result_val_test(y_test, predicted_value3, data_set)  
    
    Draw_RandomForest_Result.draw_regressor_magnitude_error(y_test_magnitude, predicted_value3_magnitude, data_set)  

    # 計算預測加速度值轉換為震度的正確率
    Accuracy_final_magnitude_train = np.count_nonzero(final_y_train_magnitude==final_predicted_value1_magnitude)/len(final_y_train_magnitude)
    Accuracy_magnitude_test = np.count_nonzero(y_test_magnitude==predicted_value3_magnitude)/len(y_test_magnitude)
               
    Model_Final_Train_Mape_Result = {'TestScore_MAPE': final_train_mape, 'TestScore_MAPE_1': final_train_mape_1, 
                                     'TestScore_MAPE_2': final_train_mape_2, 'TestScore_MAPE_3': final_train_mape_3, 
                                     'TestScore_MAPE_4': final_train_mape_4, 'TestScore_MAPE_5': final_train_mape_5, 
                                     'TestScore_MAPE_6': final_train_mape_6, 'TestScore_MAPE_7': final_train_mape_7}
   
    Model_Final_Train_Mae_Result = {'TestScore_MAE': final_train_mae, 'TestScore_MAE_1': final_train_mae_1,
                                    'TestScore_MAE_2': final_train_mae_2, 'TestScore_MAE_3': final_train_mae_3,
                                    'TestScore_MAE_4': final_train_mae_4, 'TestScore_MAE_5': final_train_mae_5, 
                                    'TestScore_MAE_6': final_train_mae_6, 'TestScore_MAE_7': final_train_mae_7}
       
    Model_Test_Mape_Result = {'TestScore_MAPE': test_mape, 'TestScore_MAPE_1': test_mape_1, 
                              'TestScore_MAPE_2': test_mape_2, 'TestScore_MAPE_3': test_mape_3, 
                              'TestScore_MAPE_4': test_mape_4, 'TestScore_MAPE_5': test_mape_5, 
                              'TestScore_MAPE_6': test_mape_6, 'TestScore_MAPE_7': test_mape_7}
   
    Model_Test_Mae_Result = {'TestScore_MAE': test_mae, 'TestScore_MAE_1': test_mae_1, 
                             'TestScore_MAE_2': test_mae_2, 'TestScore_MAE_3': test_mae_3, 
                             'TestScore_MAE_4': test_mae_4, 'TestScore_MAE_5': test_mae_5, 
                             'TestScore_MAE_6': test_mae_6, 'TestScore_MAE_7': test_mae_7}
   
    return best_param_rf_reg_final, final_y_train_before_smote, Accuracy_final_magnitude_train, Model_Final_Train_Mape_Result, Model_Final_Train_Mae_Result, \
           Accuracy_magnitude_test, Model_Test_Mape_Result, Model_Test_Mae_Result
    