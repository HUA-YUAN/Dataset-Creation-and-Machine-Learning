# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:00:59 2025

@author: stanl
"""

# 外加模組
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


# 用來區分出訓練集中的真實數據與虛擬數據
def check_train_class(train_index, y_train_fold, predicted_value1): 
    train_class = []
    train_class0 = []
    train_class1 = []
    predicted_value1_class0 = []
    predicted_value1_class1 = []
            
    for n in range(len(train_index)):
        train_class1.append(y_train_fold[n])
        predicted_value1_class1.append(predicted_value1[n])
            
    for m in range(len(y_train_fold)-len(train_index)):
        train_class0.append(y_train_fold[len(train_index) + m])
        predicted_value1_class0.append(predicted_value1[len(train_index) + m])

    return train_class, train_class0, train_class1, predicted_value1_class0, predicted_value1_class1



# 繪製模型的預測結果(交叉驗證中的訓練集與驗證集)
def draw_regressor_result_train(y_train_fold, predicted_value1, train_class0, train_class1, predicted_value1_class0, predicted_value1_class1, data_set): 
    fig, ax = plt.subplots(dpi=150)
    ax.scatter(train_class1, predicted_value1_class1, color = '#88c999')
    ax.scatter(train_class0, predicted_value1_class0, color = '#EF4026')
    #ax.plot([y_train_fold.min(), y_train_fold.max()], [y_train_fold.min(), y_train_fold.max()], 'k-', lw=2)
    ax.plot([0, 350], [0, 350], 'k-', lw=2)
    plt.legend(['$\mathregular{R^{2}}$: ' + str(round(r2_score(train_class1, predicted_value1_class1),2))], loc='best')

    if data_set == "train":
        plt.title('Prediction Results for Acceleration \n(Cross-Validation Training)',fontsize=20, fontweight='bold')
    
    if data_set == "final_train":    
        plt.title('Prediction Results for Acceleration \n(Training)',fontsize=20, fontweight='bold')
  
    plt.xlim(0,350) 
    plt.ylim(0,350)     
    ax.set_xlabel('Observed Acceleration',fontsize=16)
    ax.set_ylabel('Predicted Acceleration',fontsize=16)
    plt.show()
    
 
# 繪製模型的預測結果(訓練集與測試集)    
def draw_regressor_result_val_test(y_fold, predicted_value, data_set): 
    fig, ax = plt.subplots(dpi=150)
    ax.scatter(y_fold, predicted_value, color = '#88c999')
    #ax.plot([y_val_fold.min(), y_val_fold.max()], [y_val_fold.min(), y_val_fold.max()], 'k-', lw=2)
    ax.plot([0, 350], [0, 350], 'k-', lw=2)
    plt.legend(['$\mathregular{R^{2}}$: ' + str(round(r2_score(y_fold, predicted_value),2))], loc='best')
    
    if data_set == "val":
        plt.title('Prediction Results for Acceleration \n(Cross-Validation Validation)',fontsize=20, fontweight='bold')    
    
    if data_set == "test":
        plt.title("Prediction Results for Acceleration \n(Testing)",fontsize=20, fontweight='bold')
    
    plt.xlim(0,350) 
    plt.ylim(0,350)     
    ax.set_xlabel('Observed Acceleration',fontsize=16)
    ax.set_ylabel('Predicted Acceleration',fontsize=16)
    plt.show()
    

# 繪製將模型預測加速度值轉換為震度的結果(交叉驗證中的訓練集與驗證集)
# 繪製將模型預測加速度值轉換為震度的結果(最終訓練集與驗證集)
def draw_regressor_magnitude_error(y_magnitude, x_magnitude, data_set):  
    magnitude_error_1 = []
    magnitude_error_2 = []
    magnitude_error_3 = []
    magnitude_error_4 = []
    magnitude_error_5 = []
    magnitude_error_6 = []
    magnitude_error_7 = []
        
    magnitude_error = x_magnitude - y_magnitude
    magnitude_error = list(magnitude_error)

    for k in range(len(y_magnitude)):
        if y_magnitude[k] == 7:
            magnitude_error_7.append(magnitude_error[k])
            
        elif y_magnitude[k] == 6:
            magnitude_error_6.append(magnitude_error[k])
            
        elif y_magnitude[k] == 5:
            magnitude_error_5.append(magnitude_error[k])
            
        elif y_magnitude[k] == 4:
            magnitude_error_4.append(magnitude_error[k])
            
        elif y_magnitude[k] == 3:
            magnitude_error_3.append(magnitude_error[k])
            
        elif y_magnitude[k] == 2:
            magnitude_error_2.append(magnitude_error[k])
            
        elif y_magnitude[k] == 1:
            magnitude_error_1.append(magnitude_error[k])
    
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 使用中文字體
    col_names = ["低估3級","低估2級","低估1級","判斷正確","高估1級","高估2級","高估3級"]
    #col_names = ["-3","-2","-1","0","+1","+2","+3"]
    
    error71 = magnitude_error_7.count(-3)
    error72 = magnitude_error_7.count(-2)
    error73 = magnitude_error_7.count(-1)
    error74 = magnitude_error_7.count(0)
    error75 = 0
    error76 = 0
    error77 = 0
    col_error_numbers_7 = [error71, error72, error73, error74, error75, error76, error77]

    error61 = magnitude_error_6.count(-3)
    error62 = magnitude_error_6.count(-2)
    error63 = magnitude_error_6.count(-1)
    error64 = magnitude_error_6.count(0)
    error65 = magnitude_error_6.count(1)
    error66 = 0
    error67 = 0
    col_error_numbers_6 = [error61, error62, error63, error64, error65, error66, error67]

    error51 = magnitude_error_5.count(-3)
    error52 = magnitude_error_5.count(-2)
    error53 = magnitude_error_5.count(-1)
    error54 = magnitude_error_5.count(0)
    error55 = magnitude_error_5.count(1)
    error56 = magnitude_error_5.count(2)
    error57 = 0
    col_error_numbers_5 = [error51, error52, error53, error54, error55, error56, error57]

    error41 = magnitude_error_4.count(-3)
    error42 = magnitude_error_4.count(-2)
    error43 = magnitude_error_4.count(-1)
    error44 = magnitude_error_4.count(0)
    error45 = magnitude_error_4.count(1)
    error46 = magnitude_error_4.count(2)
    error47 = magnitude_error_4.count(3)
    col_error_numbers_4 = [error41, error42, error43, error44, error45, error46, error47]

    error31 = magnitude_error_3.count(-3)
    error32 = magnitude_error_3.count(-2)
    error33 = magnitude_error_3.count(-1)
    error34 = magnitude_error_3.count(0)
    error35 = magnitude_error_3.count(1)
    error36 = magnitude_error_3.count(2)
    error37 = magnitude_error_3.count(3)
    col_error_numbers_3 = [error31, error32, error33, error34, error35, error36, error37]

    error21 = 0
    error22 = magnitude_error_2.count(-2)
    error23 = magnitude_error_2.count(-1)
    error24 = magnitude_error_2.count(0)
    error25 = magnitude_error_2.count(1)
    error26 = magnitude_error_2.count(2)
    error27 = magnitude_error_2.count(3)
    col_error_numbers_2 = [error21, error22, error23, error24, error25, error26, error27]

    error11 = 0
    error12 = 0
    error13 = 0
    error14 = magnitude_error_1.count(0)
    error15 = magnitude_error_1.count(1)
    error16 = magnitude_error_1.count(2)
    error17 = magnitude_error_1.count(3)
    col_error_numbers_1 = [error11, error12, error13, error14, error15, error16, error17]

    # hard-coded data
    dist_lst = [col_error_numbers_1, col_error_numbers_2, col_error_numbers_3, col_error_numbers_4,
                col_error_numbers_5, col_error_numbers_6, col_error_numbers_7]
    lst = np.array(dist_lst)
    
    legend_name = ["震度1級", "震度2級", "震度3級", "震度4級", "震度5級","震度6級", "震度7級"]
    # defined colors
    colors = ["palegreen", "lime", "yellow", "darkorange","red", "tab:brown", "darkviolet"]
    
    i = 0
    fig, ax = plt.subplots(dpi=150)
    
    if data_set == "train":
        plt.title('Prediction Results for Intensity \n(Cross-Validation Training)', fontsize=20, fontweight='bold')
    if data_set == "val":
        plt.title('Prediction Results for Intensity \n(Cross-Validation Validation)', fontsize=20, fontweight='bold')
        
    if data_set == "final_train":
        plt.title('Prediction Results for Intensity \n(Training)', fontsize=20, fontweight='bold')  
    if data_set == "test":
        plt.title('Prediction Results for Intensity \n(Testing)', fontsize=20, fontweight='bold')
        
    ax.barh(col_names, lst[0], color=colors[0])
    sum_arr = lst[0]
    for data in lst[1:]:
        i += 1
        ax.barh(col_names, data, left=sum_arr, color=colors[i])
        sum_arr += data

    plt.legend(legend_name, ncol=2) # 顯示圖例 
    for c in ax.containers:
        # Create a new list of labels
        labels = [a if a else "" for a in c.datavalues]
        ax.bar_label(c, labels=labels, label_type="center")
    ax.set_xlabel('Number of Data',fontsize=16)     
    ax.set_ylabel('Prediction Error for Intensity',fontsize=16)
    plt.show()
    
    
# 請使用 matplotlib==3.7.3 更新之版本有顯示異常的問題
def draw_classifier_confusion_matrix(data_cm, data_set):
    #classes_magnitude = ('2', '3', '4', '5', '6')
    classes_magnitude = ('震度2級', '震度3級', '震度4級', '震度5級', '震度6級')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 使用中文字體
           
    labels = pd.DataFrame(data_cm).applymap(lambda v: f"{v}" if v!=0 else f"")
    plt.figure(figsize=(7,5),dpi=150)
    sns.heatmap(data_cm, annot=labels, fmt='s', xticklabels=classes_magnitude, yticklabels=classes_magnitude, linewidths=0.1,cmap=plt.cm.Blues )
    
    if data_set == "train":
        plt.title('Prediction Results for Intensity \n(Cross-Validation Training)', fontsize=20, fontweight='bold')
    if data_set == "val":
        plt.title('Prediction Results for Intensity \n(Cross-Validation Validation)', fontsize=20, fontweight='bold')
        
    if data_set == "final_train":
        plt.title('Prediction Results for Intensity \n(Training)', fontsize=20, fontweight='bold')  
    if data_set == "test":
        plt.title('Prediction Results for Intensity \n(Testing)', fontsize=20, fontweight='bold')

    plt.xlabel('Predicted Magnitude',fontsize=16)
    plt.ylabel('Observed Magnitude',fontsize=16)
    plt.show()   

    