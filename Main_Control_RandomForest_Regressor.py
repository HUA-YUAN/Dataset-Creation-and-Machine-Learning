
# 外加模組
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# 自訂函式
import Feature_Distribution
import Classify_Earthquake_Magnitude
import RandomForest_Regressor
import RandomForest_Gridsearch
 

   
    
if __name__ == '__main__':
    
   # Calling the data
   dataset = pd.read_csv("D:\Desktop\研究計畫資料\test\資料集二特徵值C組.csv")

   feature = "C"
   '''
       A: 特徵值A組(統計特徵組合)
       B: 特徵值B組(能量特徵組合)
       C: 特徵值C組(合併後特徵組合)
   '''

   if feature == "A":
       data = dataset.iloc[:,0:5]
       target = dataset.iloc[:,5]
       
   if feature == "B":
       data = dataset.iloc[:,0:4]
       target = dataset.iloc[:,4]
       
   if feature == "C":
       data = dataset.iloc[:,0:7]
       target = dataset.iloc[:,7]
   
   # Input location of the data
   data = data.values
   data = data.astype('float32')
   
   # Output location of the data
   target = target.values
   target = target.astype('float32')

   # 特徵分布圖(特徵值 & 最大地動加速度)
   Feature_Distribution.draw_dataset_feature_distribution(dataset, target, feature)
   
   smote_mode = "no"
   '''
       yes: 使用smote
       no: 不使用
   '''
   
   output_mode = "pga"
   '''
       pga: 輸出為pga
       pga^2: 輸出為pga取二次方
       log(pga): 輸出為pga取log
   '''

   # Create StratifiedKFold object
   stratified_n_splits = 9
   stratified_random_state = 7
   stratified = StratifiedKFold(n_splits=stratified_n_splits , shuffle=True, random_state=stratified_random_state)
  
   '''
       交叉驗證參數含義：
       estimator: 設定需要優化的模型種類
       param_grid: 設定需要優化的使用參數值
       n_splits: 將資料分割為 K 等分
       n_repeats: 反覆執行次數
       random_state: 需要 shuffle 為 True 才需要設定，功能如同 seed
       scoring: 選擇誤差的計算方式
           "r2": 殘差平方和
           "neg_mean_absolute_percentage_error": MAPE 百分誤差
           "neg_mean_squared_error": MSE 均方誤差
           "neg_mean_absolute_error": MAE 平均絕對誤差
   '''
   
   for_stratify_magnitude = Classify_Earthquake_Magnitude.classify_magnitude_for_dataset(target)
   
   # 於所有資料中劃分出訓練集與測試集 (Training and Testing)
   x_train_val, x_test, y_train_val, y_test = train_test_split(data, target, test_size=0.1 ,
                                                       random_state=10, shuffle=True, stratify=for_stratify_magnitude)
 
   for_stratified_magnitude = Classify_Earthquake_Magnitude.classify_magnitude_for_dataset(y_train_val)
   
   
   # 進入網格搜索法模式
   mode = "gridsearch"
   # 得出最佳的超參數和迭代變化紀錄
   best_parameters, gridsearch_history = RandomForest_Gridsearch.regressor_gridsearch(x_train_val, y_train_val, stratified, for_stratified_magnitude, mode, feature, output_mode, smote_mode)
   
   # 進入使用最佳超參數模式
   mode = "best_parameters"
   # 得出最佳的超參數
   seed = best_parameters['seed']
   max_depth = best_parameters['max_depth']
   n_estimators = best_parameters['n_estimators']
   
   # 使用最佳的超參數得出交叉驗證結果
   best_param_rf_reg, Model_Train_Mape_Result, Model_Train_Mae_Result, Model_Validation_Mape_Result, Model_Validation_Mae_Result, Accuracy_Magnitude\
                    = RandomForest_Regressor.regressor_cross_validation(seed, x_train_val, y_train_val, stratified, for_stratified_magnitude, mode, max_depth, n_estimators, feature, output_mode, smote_mode)
   
   # 使用最佳的超參數套用在原先的訓練集與測試集
   best_param_rf_reg_final, final_y_train_before_smote, Accuracy_final_magnitude_train, Model_Final_Train_Mape_Result, Model_Final_Train_Mae_Result, \
   Accuracy_magnitude_test, Model_Test_Mape_Result, Model_Test_Mae_Result \
       = RandomForest_Regressor.regressor_testing(seed, max_depth, n_estimators, x_test, y_test, x_train_val, y_train_val, output_mode, smote_mode, feature)
    
    

