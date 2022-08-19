# NASA - Nearest Earth Objects 利用機器學習模型判斷小行星特徵值與對地球危害程度預估
## 發想動機
本報告主題發想自Netflix 2021年製作電影「千萬別抬頭 Don’t Look Up」，劇中藉由彗星撞地球的危機影射現代社會中的政治經濟問題。 其中彗星對地球是否會對地球造成危害產生了廣大的討論，由此我 們發想出可以利用NASA Open API 進行機器學習的模型實作。
![](https://i.imgur.com/vE0RNJt.png)
圖、千萬別抬頭宣傳海報
## 資料庫介
 在資料庫中總共包含了三項類別型資料、五項連續型資料及兩項布林值
## 問題定義 
在接下來的分析方法中我們計畫達成兩大目標，分別是:
1. 利用視覺化方法分類並找到個變數之間的關聯 
2. 利用機器學習模型實踐對地球危害小行星的辨別
## 資料前處理
### 資料分類
使用 𝑝𝑎𝑛𝑑𝑎𝑠.𝑖𝑠𝑛𝑢𝑙𝑙( ).𝑠𝑢𝑚 函式統計資料庫中資料缺失的狀況，在這裡並無資料缺失因此可以直接進入下一階段的分析。
![](https://i.imgur.com/IeG36Bi.png)
id, name, orbiting_body, sentry_object等四項類別型變數在特 徵工程上沒有明顯的價值，因此在後續的分析忽略不採計。
同時在分類時將hazardous的布林值True, False轉為數值型態 1, 0方便操作。並將變數裡的est_diameter_min (Minimum Estimated Diameter in Kilometres), est_diameter_max (Maximum Estimated Diameter in Kilometres)平均，創造一全 新變數欄位diameter_mean。
 
## 資料視覺化分析
1. 將所有連續型變數利用pairplot的形式同步繪製觀察其相關性。大致上皆為隨機散佈，可以觀察到diameter(小行星直徑)和absolute magnitude(絕對星等)呈現負相關。
![](https://i.imgur.com/W1ds1g5.png)

2. 皮爾森積動差相關係數: 將各連續性變數的相關係數以熱圖方式呈現
![](https://i.imgur.com/qroEZ4J.png)

3. 核密度估計(Kernel Density Estimation):
核密度估計可以看出隨機變數落在特定值的可能性
套用seaborn中kde.plot()的方法實踐核密度作圖，這裡討論relevative velocity, diameter, miss distance, absolute magnitude等四個變數，並以hazardous作為hue分類  
![](https://i.imgur.com/wGTJeIX.png)![](https://i.imgur.com/ALHSVNf.png)
左圖、具威脅小行星移動速度略快於不具威脅性類別 右圖、資料集中小行星大小分佈集中  
![](https://i.imgur.com/eFVc5F7.png)![](https://i.imgur.com/2OeKouJ.png)  
左圖、距離與威脅性較無明顯關聯 右圖、具威脅性小行星絕對星等集中於20左右 又，絕對星等和小行星直徑大小成負相關關係(如下圖)，因此我們可以知道具威脅性小行星集中在特定大小類型
![](https://i.imgur.com/ppdvW6r.png)  
絕對星等與小行星平均直徑散佈圖

## 機器學習模型實作 
總共選用七種不同的機器學習模型，分別為:
1. Decision Tree
2. XG Boost
3. KNN
4. Random Forest
5. Gaussian Naive Bayes 6. SVC
7. Logistic Regression
以Decision Tree為例，示範程式碼的建立方法
```python
# 匯入決策樹模型
from sklearn.tree import DecisionTreeClassifier 
# 匯入準確度計算工具
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# 創造決策樹模型
# 設定最佳化方法為 Gini Index
# 設定最大深度為 2
# 設定最多葉子個數為 4
model = DecisionTreeClassifier(                 
    criterion='gini',                           
    max_depth=2,                                
    max_leaf_nodes=2 ** 2
)                                     
# 訓練決策樹模型
model.fit(x, y)                     

# 確認模型是否訓練成功
pred_y = model.predict(x)                 
# 計算準確度、f1 score
Acc_DTC = accuracy_score(y, pred_y)   
f1_DTC = f1_score(y, pred_y)        

# 輸出準確度
print('accuracy: {}\n'.format(Acc_DTC)+
      'f1 score: {}'.format(f1_DTC))
```
![](https://i.imgur.com/GNz65A6.png)
深度為2 最大葉子數為4 的Decision Tree視覺化模型圖形 

## 模型結果比較
- error(%)
- DTC: Decision Tree Classifier
- KNN: K Neighbors Classifier
- RF : Random Forest Classifier
- GNB: Gaussian Naive Bayes
- SVC: Support Vector Machine(Regression)
- LR : Logistic Regression
Average Valid Accuracy為我們使用K次交叉驗證工具(K-Fold Cross-Validation)，在 K-Fold 的方法中我們會將資料切分為 K 等份 ，以K-1份作為訓練資料，剩下1份作為驗證資料並在訓練完畢後驗 證預測計算其誤差值。
在上述七種模型的測試當中，Valid Accuracy與Train Accuracy並無 太大區別，可以確定在模型試驗中並沒有過度擬合的問題。
  


|  | DTC | XG Boost |KNN|RF|GNB|SVC|LR|
| --- | --- | --- | --- | --- | --- | --- | --- |
| Train Accuracy | 0.9125 | 0.9133|0.9179|0.9129|0.897|0.903|0.9027|
| Valid Accuracy |0.9124| 0.9128|0.8812|0.9125|0.897|0.902|0.9026|
| error(%) | 0.011 | 0.055|3.99|0.0438|0.011|0.022|0.011|

我們也觀察到在訓練準確度(Train Accuracy)中支援向量機(SVC) 和邏輯回歸(Logistice Regression)模型的準確度非常相近，推測是 因為兩者皆是採取回歸分析的方法進行訓練，與其他分類學習法不 同，因此結果較為接近。
## 問題與討論
除了使用Scikit Learn中的accuracy_score()函式進行模型準確度的 評分，我們還引入f1_score評估機器學習模型成效。
F1 Score的定義如下:
$$F1\quad Score=\frac{2\times Precision\times Recall}{Precision +Recall}$$
$$Precision=\frac{TP}{TP+FP}\quad Recall=\frac{TP}{TP+FN}$$


   是基於混淆矩陣所定義出的分數，為召回率(Recall)和準確率 (Precision)的 ，跑分結果如下表:
然而在先前accuracy_score獲得高分的模型在f1_score評斷下的表 現卻不甚理想，重新檢視資料庫本身我們發現了一項問題:作為輸 出Y值的變數hazardous除了本身為布林值僅有真假兩項，資料集本 身更有數據不平衡的問題。對地球有威脅性的資料僅佔整體數據 10%。 若在未進行前處理的狀況下直接對其進行訓練，就會發生上述以預 測Y值和實際Y值進行評斷的accuracy_score準確度獲得高分; 模型本身的f1_score分數卻不理想的狀況。

 解決這項問題我們預計在未來有機會操作時先以Random Under/Over Sampling進行資料前處理，避免相同情形產生。 同時也藉此機會了解到資料集本身資料分佈的優劣也在機器學習領 域佔有舉足輕重的地位。
## 完整程式碼 
[colab](https://colab.research.google.com/drive/1kVk7loF3_xIRCgs__stndyRz-XIAgzXw?usp=sharing)

## 參考資料
- [JetPropulsionLab](
https://cneos.jpl.nasa.gov/ca/)
- NASAOpenAPI
- [NASA-NearestEarthObjects](https://www.kaggle.com/code/elnahas/nasa-nearest-earth-objects/notebook)
- [N.E.O.ClassificationwithDecisionTree](https://www.kaggle.com/code/johnyoungsorensen/n-e-o-classification-with-decision-tree/data)
- [全民瘋AI系列2.0](https://ithelp.ithome.com.tw/users/20107247/ironman/4723)
- [Precision,Recall,F1-scorer簡單介紹](https://medium.com/nlp-tsupei/precision-recall-f1-score%E7%B0%A1%E5%96%AE%E4%BB%8B%E7%B4%B9-f87baa82a47)
- [白話解釋核密度估計(KernelDensityEstimation)](https://medium.com/qiubingcheng/%E7%99%BD%E8%A9%B1%E8%A7%A3%E9%87%8B%E6%A0%B8%E5%AF%86%E5%BA%A6%E4%BC%B0%E8%A8%88-kernel-density-estimation-18c4913f0b6a)
    