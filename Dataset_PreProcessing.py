# -*- coding: utf-8 -*-
"""
Created on Thu May  1 13:45:01 2025

@author: stanl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.integrate as spi



plt.style.use("ggplot")
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
plt.rcParams['axes.unicode_minus'] = False


# 判斷P波到達時間   
def automatically_picking_p_time(a, dt, n, sta, lta, p_value):
    # 計算P波的長短時平均
    n_sta = int(sta/dt)
    n_lta = int(lta/dt)
    a1 = abs(a)

    a_sta = [0] * n  
    for i in range(n_sta,n):           
            a_sta[i] = (sum(a1[i-n_sta:i-1]))/n_sta

    a_lta = [0] * n  
    for i in range(n_lta,n):          
            a_lta[i] = (sum(a1[i-n_lta:i-1]))/n_lta

    r = [0] * n  
    for i in range(n_lta,n):
            r[i] = a_sta[i]/a_lta[i]

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
    

def draw_p_arrive_test(t, p_arrive, a_u, r):
    fig, ax = plt.subplots(2,1,dpi=150, figsize = (8,4.5))
    ax[0].set_title('U-D acceleration and STA/LTA ratio')
    ax[0].plot(t,a_u)
    ax[0].set_ylabel(r"U-D acceleration")
    ax[0].grid(True)
    ax[0].set_xlim(0,t[-1])
    ax[0].axvline(x=p_arrive, ymin=0, ymax=3,color="blue")
    
    ax[1].plot(t,r)
    ax[1].set_ylabel(r"STA/LTA")
    ax[1].grid(True)
    ax[1].set_xlim(0,t[-1])
    ax[1].axvline(x=p_arrive, ymin=0, ymax=3,color="blue")
    
    plt.xlabel(r"time (s)")
    plt.tight_layout()
    #plt_name = j.split("/")[-1].replace(".txt","") + "_p" + str(p_arrive) + ".png"
    #plt.savefig(plt_name)
    plt.show()
    
    

# 將加速度值進行積分取得速度與位移值
def integrate_acceleration(a, dt, n):
    v = [0] * n
    v0 = 0
    v1 = spi.cumtrapz(a,dx=dt,initial=v0)
    for i in range(1, n):
        v[i] = v1[i]-v1[i-1]
    
    d = [0] * n
    d0=0
    d1 = spi.cumtrapz(v,dx=dt,initial=d0)  
    for i in range(1, n):
        d[i] = d1[i]-d1[i-1]
    
    cav = [0] * n
    cav1 = [0] * n
    for i in range(1,n):           
           cav1[i] = abs(v[i] - v[i-1])
    for i in range(1,n):  
           cav[i] = sum(cav1[0:i])
           
    return v, d, cav


# 繪製STA、LTA與比值變化圖 
def draw_time_a_v_t(t,a,v,cav): 
    # 設定nrows(垂直)、ncols(水平)和dpi(解析度)
    fig, ax = plt.subplots(3,1,dpi=150,figsize = (8,6))
    # ax[i]中的i為index(位置)參數
    ax[0].set_title('STA、LTA與比值變化圖')
    ax[0].plot(t,a)
    ax[0].set_ylabel(r"STA")
    ax[0].grid(True)
    ax[0].set_xlim(0,t[-1])
   
    ax[1].plot(t,v)
    ax[1].set_ylabel(r"LTA")
    ax[1].grid(True)
    ax[1].set_xlim(0,t[-1])
    
    ax[2].plot(t,cav)
    ax[2].set_ylabel(r"STA/LTA")
    ax[2].grid(True)
    ax[2].set_xlim(0,t[-1])
   
    plt.xlabel(r"時間 (s)")
    plt.tight_layout()
    # 將圖片命名並儲存為.png檔
    #plt_name = j.split("/")[-1].replace(".txt","") + "_p" + str(p_arrive) + ".png"
    #plt.savefig(plt_name)
    plt.show()
    

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

    
    return data_Pa, data_Pv, data_Pd, data_CAV_p,\
           data_Max, data_Min, data_P2P, data_Abs_Mean, data_STD
