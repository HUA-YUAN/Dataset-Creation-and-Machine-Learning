
# 外加模組
import numpy as np
import glob
import pandas as pd
import os
import time

# 自訂函式
import Dataset_PreProcessing



if __name__ == '__main__':
    # 測試執行時間用
    start_time = time.time()
    
    dirPathPattern = r"D:\Desktop\研究計畫資料\test\*.txt" # 讀取地震資料 
    # 想查詢的資料夾位置 *.後為檔案類型
    result = glob.glob(dirPathPattern)
    
    Name = []  # 數據名稱
    p_time = [] #p波到時
    Pa = []  # 加速度峰值
    Pv = []  # 速度峰值
    Pd = []  # 位移峰值
    CAV_p = []  # 累積絕對速度值
    Max = []  # 最大值
    Min = []  # 最小值
    P2P = []  # 峰間值
    Abs_Mean = []  # 絕對平均值
    STD = []  # 標準差
    pga_all = []  # 最大加速度峰值
    
    
    for j in result:
        data = np.loadtxt(j)
        #data = np.loadtxt(j,skiprows=15)
        #skiprows 指跳過的行數
        dt = 0.005 # 時間間隔  
        a = data[:,1]  # 垂直向(P波)
        n = len(a)  # 數據長度
        t = np.linspace(0.0,dt*(n-1),n)
        
        file_name_all = os.path.basename(j)#取得檔案名稱含副檔名
        file_name = os.path.splitext(file_name_all)[0] #取得檔案名稱不含副檔名
        Name.append(file_name) 
        
        # 計算pga
        a2 = data[:,2] # 南北向
        a3 = data[:,3] # 東西向
        a_all = np.sqrt((abs(a)**2)+(abs(a2)**2)+(abs(a3)**2))
        pga = max(a_all)
        pga_all.append(pga)
            
        # 自動偵測P波方法之設定參數
        sta = 1.0
        lta = 5.0
        p_value = 2.0
        a_sta, a_lta, r, p_arrive = Dataset_PreProcessing.automatically_picking_p_time(a, dt, n, sta, lta, p_value) 
        p_time.append(p_arrive)
        
        # 繪製STA、LTA與比值變化圖
        Dataset_PreProcessing.draw_time_sta_lta(t, a_sta, a_lta, r)
        # 繪製P波加速度圖與比值變化圖 
        Dataset_PreProcessing.draw_p_arrive_test(t, p_arrive,a , r)
        
        # 將加速度值進行積分取得速度與位移值
        v, d, cav = Dataset_PreProcessing.integrate_acceleration(a,dt,n)
 
        # 濾波 
        fl = 0.075
        fh = 10
        btype = "highpass"
        '''
        濾波參數含義：
        x: 信號序列
        dt: 信號的採樣時間間隔
        fl: 濾波截止頻率（低頻）
        fh: 濾波截止頻率（高頻）
        order: 濾波器的階數
        btype: 濾波器類型
            "lowpass": 低通（保留低於fh的頻率成分）
            "highpass": 高通（保留高於fl的頻率成分）
            "bandpass": 帶通（保留fl~fh之間的頻率成分）
            "bandstop": 帶阻（過濾fl~fh之間的頻率成分）
        '''
        v_f = Dataset_PreProcessing.filter_wave(v, dt, fl, fh, btype)
        d_f = Dataset_PreProcessing.filter_wave(d, dt, fl, fh, btype)
        
        # 計算所有特徵值
        data_Pa, data_Pv, data_Pd, data_CAV_p, data_Max, data_Min, data_P2P, data_Abs_Mean, data_STD = Dataset_PreProcessing.calculate_feature(dt, a, p_arrive, v_f, d_f, cav)
        
        Pa.append(data_Pa)
        Pv.append(data_Pv)
        Pd.append(data_Pd)
        CAV_p.append(data_CAV_p)

        Max.append(data_Max)       
        Min.append(data_Min)
        P2P.append(data_P2P)
        Abs_Mean.append(data_Abs_Mean)
        STD.append(data_STD)
    
    dic = {"File_Name":Name,"p_time":p_time,"Max":Max,"Min":Min,"P2P":P2P,"Abs_Mean":Abs_Mean,"STD":STD,"Pa":Pa,"Pv":Pv,"Pd":Pd,"CAV":CAV_p,"Accleration":pga_all}
    df = pd.DataFrame(dic)
    
    # 將所有資料儲存成csv檔
    df.to_csv('earthquake_eigenvalue_dataframe2.csv')
    
    # 測試執行時間用
    end_time = time.time()
    execution_time = end_time - start_time
    print("程式執行時間：", execution_time, "秒")
    
        
    



    
