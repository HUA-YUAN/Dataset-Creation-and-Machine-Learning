# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:35:27 2025

@author: stanl
"""

# 外加模組
from matplotlib import pyplot as plt

# 自訂函式
import RandomForest_Regressor
import RandomForest_Classifier


# 網格搜索法的迭代過程
def regressor_gridsearch(x_train_val, y_train_val, stratified, for_stratified_magnitude, mode, feature, output_mode, smote_mode):
    best_score = 0.0
    
    gridsearch_val_mape = []
    gridsearch_val_mae = []
    gridsearch_max_depth = []
    gridsearch_n_estimators = []
    gridsearch_seed = []
    
    for seed in range(7,8,1):
        
        for max_depth in range(6,7,1):
            
            gridsearch_draw_n_estimators = []
            gridsearch_draw_train_mape = []
            gridsearch_draw_train_mae = []
            gridsearch_draw_val_mape = []
            gridsearch_draw_val_mae = []      
            
            #for n_estimators in [25, 50, 75, 100, 150, 250, 300]:
            for n_estimators in range(112,113,1):
 
                val_score1, val_score2, train_score1, train_score2 = RandomForest_Regressor.regressor_cross_validation(seed, x_train_val, y_train_val, stratified, for_stratified_magnitude, 
                                                                                  mode, max_depth, n_estimators, feature, output_mode, smote_mode)
           
                gridsearch_val_mape.append(val_score1)
                gridsearch_val_mae.append(val_score2)
                gridsearch_max_depth.append(max_depth)
                gridsearch_n_estimators.append(n_estimators)
                gridsearch_seed.append(seed)
                
                gridsearch_draw_train_mape.append(train_score1)
                gridsearch_draw_train_mae.append(train_score2)
                gridsearch_draw_val_mape.append(val_score1)
                gridsearch_draw_val_mae.append(val_score2)
                gridsearch_draw_n_estimators.append(n_estimators)
      
                if best_score == 0.0:
                    #best_score = val_score1
                    best_score = val_score2
                    best_parameters = {"seed":seed,"max_depth":max_depth,"n_estimators":n_estimators,"mape": val_score1,"mae": val_score2}
                if val_score2 < best_score:
                    #best_score = val_score1
                    best_score = val_score2
                    best_parameters = {"seed":seed,"max_depth":max_depth,"n_estimators":n_estimators,"mape": val_score1,"mae": val_score2}
            
            draw_regressor_gridsearch_mape(seed, max_depth, gridsearch_draw_n_estimators, gridsearch_draw_train_mape, gridsearch_draw_val_mape)
            draw_regressor_gridsearch_mae(seed, max_depth, gridsearch_draw_n_estimators, gridsearch_draw_train_mae, gridsearch_draw_val_mae)
    
    gridsearch_history = {'seed':seed, 'max_depth':gridsearch_max_depth,'n_estimators': gridsearch_n_estimators,
                          'mape_score': gridsearch_val_mape, 'mae_score': gridsearch_val_mae,}
    
    return best_parameters, gridsearch_history


# 繪製網格搜索法的迭代MAPE變化圖
def draw_regressor_gridsearch_mape(seed, max_depth, gridsearch_draw_n_estimators, gridsearch_draw_train_mape, gridsearch_draw_val_mape):
    fig, ax = plt.subplots(dpi=150)
    plt.plot(gridsearch_draw_n_estimators, gridsearch_draw_train_mape, 'b', label = 'Training Mean Mape', color='#6495ED')
    plt.plot(gridsearch_draw_n_estimators, gridsearch_draw_val_mape, 'b', label = 'Validation Mean Mape', color='coral')
    plt.title('Gridsearch History (Mean Mape)')
    plt.xlabel('n_estimators')
    plt.ylabel('Mean Mape')
    plt.legend(title='seed = '+ str(seed) + ' ,max_depth = '+ str(max_depth))
    #plt.grid()
    plt.xticks()
    plt.show()
    

# 繪製網格搜索法的迭代MAE變化圖
def draw_regressor_gridsearch_mae(seed, max_depth, gridsearch_draw_n_estimators, gridsearch_draw_train_mae, gridsearch_draw_val_mae):
    fig, ax = plt.subplots(dpi=150)
    plt.plot(gridsearch_draw_n_estimators, gridsearch_draw_train_mae, 'b', label = 'Training Mean Mae', color='#6495ED')
    plt.plot(gridsearch_draw_n_estimators, gridsearch_draw_val_mae, 'b', label = 'Validation Mean Mae', color='coral')
    plt.title('Gridsearch History (Mean Mae)')
    plt.xlabel('n_estimators')
    plt.ylabel('Mean Mae')
    plt.legend(title='seed = '+ str(seed) + ' ,max_depth = '+ str(max_depth))
    #plt.grid()
    plt.xticks()
    plt.show()
    

def classifier_gridsearch(x_train_val, y_train_val, stratified, for_stratified_magnitude, mode, feature):
    best_score = 0.0
    
    gridsearch_train_accuracy = []
    gridsearch_val_accuracy = []
    gridsearch_max_depth = []
    gridsearch_n_estimators = []
    gridsearch_seed = []
    
    for seed in range(7,8,1):
        
        for max_depth in range(9,10,1):
            
            gridsearch_draw_train_accuracy = []
            gridsearch_draw_val_accuracy = []
            gridsearch_draw_n_estimators = []
            gridsearch_draw_max_depth = []
            
            #for n_estimators in [25, 50, 75, 100, 150, 250, 300]:
            for n_estimators in range(7,8,1):
 
                train_score1, val_score1 = RandomForest_Classifier.classifier_cross_validation(seed, x_train_val, y_train_val, stratified, for_stratified_magnitude, mode, max_depth, n_estimators, feature)
           
                gridsearch_train_accuracy.append(train_score1)
                gridsearch_val_accuracy.append(val_score1)
                gridsearch_max_depth.append(max_depth)
                gridsearch_n_estimators.append(n_estimators)
                gridsearch_seed.append(seed)
      
                gridsearch_draw_train_accuracy.append(train_score1)
                gridsearch_draw_val_accuracy.append(val_score1)
                gridsearch_draw_max_depth.append(max_depth)
                gridsearch_draw_n_estimators.append(n_estimators)
      
                if best_score == 0.0:
                    best_score = val_score1
                    best_parameters = {"seed":seed,"max_depth":max_depth,"n_estimators":n_estimators,"train_accuracy": train_score1,"val_accuracy": val_score1}
                if val_score1 > best_score:
                    best_score = val_score1
                    best_parameters = {"seed":seed,"max_depth":max_depth,"n_estimators":n_estimators,"train_accuracy": train_score1,"val_accuracy": val_score1}

            draw_gridsearch_magnitude_accuracy(seed, max_depth, gridsearch_draw_n_estimators, gridsearch_draw_train_accuracy, gridsearch_draw_val_accuracy)
    
    gridsearch_history = {'seed':seed, 'max_depth':gridsearch_max_depth,'n_estimators': gridsearch_n_estimators,'train_accuracy': gridsearch_train_accuracy,'val_accuracy': gridsearch_draw_val_accuracy}
    
    return best_parameters, gridsearch_history


def draw_gridsearch_magnitude_accuracy(seed, max_depth, gridsearch_draw_n_estimators, gridsearch_draw_train_accuracy, gridsearch_draw_val_accuracy):
    fig, ax = plt.subplots(dpi=150)
    plt.plot(gridsearch_draw_n_estimators, gridsearch_draw_train_accuracy, 'b', label = 'Training Accuracy', color='#6495ED')
    plt.plot(gridsearch_draw_n_estimators, gridsearch_draw_val_accuracy, 'b', label = 'Validation Accuracy', color='coral')
    plt.title('Validation Gridsearch History (Magnitude Accuracy)')
    plt.xlabel('n_estimators')
    plt.ylabel('Magnitude Accuracy')
    plt.legend(title='seed = '+ str(seed) + ' ,max_depth = '+ str(max_depth))
    #plt.grid()
    plt.show()