# -*- coding: utf-8 -*-
"""
Created on Thu May  1 13:45:01 2025

@author: stanl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal



# 判斷P波到達時間   
def automatically_picking_p_time(a, dt, n, sta, lta, p_value):
    # 計算P波的長短時平均
    n_sta = int(sta/dt)
    n_lta = int(lta/dt)
    a1 = abs(a)
    
    a_sta = []
    a_lta = []
    r = []
    
    for i in range(0,n_sta):
            a_sta.insert(i,0)
            
    for i in range(0,n_lta):
            a_lta.insert(i,0)
            r.insert(i,0)
            
    for i in range(n_sta,n):           
            a_sta1 = (sum(a1[i-n_sta:i-1]))/n_sta
            a_sta.append(a_sta1)
            
    for i in range(n_lta,n):          
            a_lta1 = (sum(a1[i-n_lta:i-1]))/n_lta
            a_lta.append(a_lta1)
    
    for i in range(n_lta,n):
            r1 = a_sta[i]/a_lta[i]
            r.append(r1)  
            
    for i in range(n_lta,n): 
        if r[i] >= p_value:
            p_arrive = i*dt
            break   
        
    return a_sta, a_lta, r, p_arrive


# 繪製STA、LTA與比值變化圖 
def draw_time_sta_lta(t,a_sta,a_lta,r): 
    # 設定nrows(垂直)、ncols(水平)和dpi(解析度)
    fig, ax = plt.subplots(3,1,dpi=150,figsize = (8,6))
    # ax[i]中的i為index(位置)參數
    ax[0].set_title('STA、LTA與比值變化圖')
    ax[0].plot(t,a_sta)
    ax[0].set_ylabel(r"STA")
    ax[0].grid(True)
    ax[0].set_xlim(0,t[-1])
   
    ax[1].plot(t,a_lta)
    ax[1].set_ylabel(r"LTA")
    ax[1].grid(True)
    ax[1].set_xlim(0,t[-1])
    
    ax[2].plot(t,r)
    ax[2].set_ylabel(r"STA/LTA")
    ax[2].grid(True)
    ax[2].set_xlim(0,t[-1])
   
    plt.xlabel(r"時間 (s)")
    plt.tight_layout()
    # 將圖片命名並儲存為.png檔
    #plt_name = j.split("/")[-1].replace(".txt","") + "_p" + str(p_arrive) + ".png"
    #plt.savefig(plt_name)
    plt.show()
    
    
# 繪製P波加速度圖與比值變化圖
def draw_p_arrive_test(t,p_arrive,a_u,r):
    # 設定nrows(垂直)、ncols(水平)和dpi(解析度)
    fig, ax = plt.subplots(2,1,dpi=150, figsize = (8,4.5))
    # ax[i]中的i為index(位置)參數
    ax[0].set_title('P波加速度圖與比值變化圖')
    ax[0].plot(t,a_u)
    ax[0].set_ylabel(r"加速度")
    ax[0].grid(True)
    ax[0].set_xlim(0,t[-1])
    ax[0].axvline(x=p_arrive, ymin=0, ymax=3,color="blue")
    
    ax[1].plot(t,r)
    ax[1].set_ylabel(r"STA/LTA")
    ax[1].grid(True)
    ax[1].set_xlim(0,t[-1])
    ax[1].axvline(x=p_arrive, ymin=0, ymax=3,color="blue")
    
    plt.xlabel(r"時間 (s)")
    plt.tight_layout()
    #plt_name = j.split("/")[-1].replace(".txt","") + "_p" + str(p_arrive) + ".png"
    #plt.savefig(plt_name)
    plt.show()
    
    

# 將加速度值進行積分取得速度與位移值
def integrate_acceleration(a, dt, n):
    v = []
    d = []
    cav = []
    
    # v = v0 + a*dt
    v = a*dt
    # d = d0 + v*dt
    d = v*dt
    
    cav0 = 0
    cav.append(cav0)
    cav2 = []
    for i in range(0,n-1):           
           cav1 = abs(v[i+1] - v[i])
           cav2.append(cav1)
           
    for i in range(1,n):  
           cav3 = sum(cav2[0:i])
           cav.append(cav3)
                   
    return v, d, cav




def filter_wave(x, dt, fl, fh, btype, order=2):
    fn = 1.0/dt/2.0  # 奈奎斯特頻率
    if btype=="lowpass":
        Wn = fh/fn
    elif btype=="highpass":
        Wn = fl/fn
    elif btype=="bandpass" or btype=="bandstop":
        Wn = (fl/fn, fh/fn)
    
    b, a = signal.butter(order, Wn, btype)
    y = signal.filtfilt(b, a, x)
    
    return y


# 根據P波到時計算所有特徵值
def calculate_feature(dt, a, p_arrive, v_f, d_f, cav): 
    eigenvalue_num = int(p_arrive/dt)
    print(eigenvalue_num)
    arrive_after3 = int(eigenvalue_num+3/dt)
    print(arrive_after3)
    print(arrive_after3-1)
    
    data_Pa = max(np.abs(a[eigenvalue_num:arrive_after3-1]))
    data_Pv = max(np.abs(v_f[eigenvalue_num:arrive_after3-1]))
    data_Pd = max(np.abs(d_f[eigenvalue_num:arrive_after3-1]))
    data_CAV_p = cav[arrive_after3-1]-cav[eigenvalue_num-1]
    
    data_Max =  np.max(a[eigenvalue_num:arrive_after3-1])
    data_Min =  np.min(a[eigenvalue_num:arrive_after3-1])
    data_P2P = data_Max-data_Min
    data_Abs_Mean = np.mean(np.abs(a[eigenvalue_num:arrive_after3-1])) 
    data_STD = np.std(a[eigenvalue_num:arrive_after3-1])

    
    return data_Pa, data_Pv, data_Pd, data_CAV_p, data_Max, data_Min, data_P2P, data_Abs_Mean, data_STD
