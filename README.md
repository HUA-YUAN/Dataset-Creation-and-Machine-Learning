<h1 align="center">
資料集與機器學習建立
</h1>
<p align = "center">
內容主要使用Python程式語言來執行，附上的.py檔案包含訊號處理，資料集建立以及機器學習演算法等。
</p>


## 內容簡介
此專案是我於學習期間發執行的一個研究計畫，研究目標是對希望能嘗試透過取得的地震加速度資料，進行訊號處理與偵測，再由機器學習領域的技術，利用取得的訊號中特徵值，來預測當次地震的最大地動加速度。

> 技術關鍵字： Python、地震 P 波擷取、機器學習、隨機森林(RandomForest)、地震震度預測


## Part 1: 資料集建立

在進行機器學習模型的訓練之前，需要先準備好指定的資料集，資料集中會有大量的原始資訊，例如目標的特徵和模式，而原始資訊的取得，可以透過系統運行時從感測器收集與目標相關的原始數據，並對數據進行特徵提取，最終建立一個可用於特定任務的資料集。


### Step 1: [原始資料取得與讀取](src/原始資料取得與讀取.md)

> 使用程式碼：  
> 1. Main_Control_Input_PreProcessing.py


### Step 2: [地震波偵測方法](src/地震波偵測方法.md)

> 使用程式碼：  
> 1. Main_Control_Pwave_Arrive_Testing.py
> 2. Dataset_PreProcessing.py


### Step 3: [訊號特徵提取](src/訊號特徵提取.md)


> 使用程式碼：  
> 1. Main_Control_Input_PreProcessing.py
> 2. Dataset_PreProcessing.py


## Part 2: 機器學習模型建立
機器學習模型是建立在演算法的基礎下，經由人類的觀察和經驗而開發出來，主要透過大量原始資訊與標準答案，從資料中自動分析獲得規律，使模型能對未知資料進行回歸或分類。此外，在資料集建立時，所取得原始資訊的品質與數量，會直接影響到機器學習模型的表現和預測結果。


### Step 4: [機器學習模型](src/機器學習模型.md)


> 使用程式碼：  
> 1. Main_Control_RandomForest_Regressor.py
> 2. Main_Control_RandomForest_Classifier.py
> 3. RandomForest_Regressor.py
> 4. RandomForest_Classifier.py

### Step 5: [交叉驗證方法](src/交叉驗證方法.md)


> 使用程式碼：  
> 1. Main_Control_RandomForest_Regressor.py
> 2. Main_Control_RandomForest_Classifier.py
> 3. RandomForest_Regressor.py
> 4. RandomForest_Classifier.py


### Step 6: [模型超參數調整](src/模型超參數調整.md)


> 使用程式碼：  
> 1. Main_Control_RandomForest_Regressor.py
> 2. Main_Control_RandomForest_Classifier.py
> 3. RandomForest_Gridsearch.py


### Step 7: [模型結果分析](src/模型結果分析.md)


> 使用程式碼：
> 1. Main_Control_RandomForest_Regressor.py
> 2. Main_Control_RandomForest_Classifier.py
> 3. RandomForest_Regressor.py
> 4. RandomForest_Classifier.py 
> 5. Calculate_Mae_Mape.py
> 6. Draw_RandomForest_Result.py


