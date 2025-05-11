
# 外加模組
import numpy as np
import glob


# 自訂函式
import Dataset_PreProcessing



if __name__ == '__main__':
    
    dirPathPattern = r"D:\Desktop\研究計畫資料\test\*.txt" # 讀取地震資料 
    # 想查詢的資料夾位置 *.後為檔案類型
    result = glob.glob(dirPathPattern)

    for j in result:
        data = np.loadtxt(j)
        #data = np.loadtxt(j,skiprows=15)
        #skiprows 指跳過的行數
        dt = 0.005 # 時間間隔  
        a = data[:,1]  # 垂直向(P波)
        n = len(a)  # 數據長度
        t = np.linspace(0.0,dt*(n-1),n)
        
        # 自動偵測P波方法之設定參數
        sta = 0.6
        lta = 3.0
        p_value = 2.0
             
        # 自動偵測P波方法之設定參數
        sta = 1.0
        lta = 5.0
        p_value = 2.0
        a_sta, a_lta, r, p_arrive = Dataset_PreProcessing.automatically_picking_p_time(a, dt, n, sta, lta, p_value) 
        
        # 繪製STA、LTA與比值變化圖
        Dataset_PreProcessing.draw_time_sta_lta(t, a_sta, a_lta, r)
        # 繪製P波加速度圖與比值變化圖 
        Dataset_PreProcessing.draw_p_arrive_test(t, p_arrive,a , r)
           

    
    
