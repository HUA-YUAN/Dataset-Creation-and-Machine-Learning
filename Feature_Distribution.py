# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:50:32 2024

@author: stanl
"""

# 外加模組
from matplotlib import pyplot as plt


# 繪製資料集中特徵值與輸出(最大地動加速度值)的分布圖
def draw_dataset_feature_distribution(dataset, target, feature): 
    if feature == "A":
        Max = dataset["Max"]  
        Min = dataset["Min"]  
        P2P = dataset["P2P"]   
        Abs_Mean = dataset["Abs_Mean"]
        STD = dataset["STD"]
    
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, Max, color = '#9900FF',label='Max',facecolors='none')
        #plt.ylim(-100,0) 
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(Max)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, Min, color = '#9900FF',label='Min',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('Min',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(Min)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, P2P, color = '#9900FF',label='P2P',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('P2P',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(P2P)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, Abs_Mean, color = '#9900FF',label='Abs_Mean',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('Abs_Mean',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(Abs_Mean)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, STD, color = '#9900FF',label='STD',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('STD',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(STD)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()

    if feature == "B":    
        Pa = dataset["Pa"]
        Pv = dataset["Pv"]
        Pd = dataset["Pd"]
        CAV = dataset["CAV"]
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, Pa, color = '#0066FF',label='Pa',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('Pa',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(Pa)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, Pv, color = '#0066FF',label='Pv',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('Pv',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(Pv)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, Pd, color = '#0066FF',label='Pd',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('Pd',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(Pd)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, CAV, color = '#0066FF',label='CAV',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('CAV',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(CAV)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
    if feature == "C":   
        P2P = dataset["P2P"]   
        Abs_Mean = dataset["Abs_Mean"]
        STD = dataset["STD"]
        Pa = dataset["Pa"]
        Pv = dataset["Pv"]
        Pd = dataset["Pd"]
        CAV = dataset["CAV"]
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, P2P, color = '#9900FF',label='P2P',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('P2P',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(P2P)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, Abs_Mean, color = '#9900FF',label='Abs_Mean',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('Abs_Mean',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(Abs_Mean)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, STD, color = '#9900FF',label='STD',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('STD',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(STD)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, Pa, color = '#0066FF',label='Pa',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('Pa',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(Pa)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, Pv, color = '#0066FF',label='Pv',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('Pv',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(Pv)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, Pd, color = '#0066FF',label='Pd',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('Pd',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(Pd)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()
        
        fig, ax = plt.subplots(dpi=150)
        ax.scatter(target, CAV, color = '#0066FF',label='CAV',facecolors='none')
        #plt.ylim(-100,0) 
        plt.title('CAV',fontsize=20, fontweight='bold')
        ax.set_xlabel('Accleration(PGA)',fontsize=16)
        ax.set_ylabel('Feature Value(CAV)',fontsize=16) 
        plt.legend(loc = 'upper center')
        plt.title('Feature Distribution',fontsize=20, fontweight='bold')
        plt.show()