# DSAI_HW1   Electricity Forecasting

<p align="center">
  <img src='img/TimeSeries.jpg'>
</p>

本作業所使用的 ``time series`` 模型為 **LSTM** ，並以前七天的資料來當作每次預測的依據，去預測接下來後七天的資料。最終目標是預測 **20210323** 到 **20210329** 為期七天的備轉容量數值。評估模型的表現是以 ``RMSE`` 為標準，而本模型的RMSE表現結果約為 **126.46** 。後面也會視覺化Validation的情況，我們可以根據線段的重疊度來大概判斷模型的準確率。

# Environment
  - **Python 3.8.3**
  - **Ubuntu 20.04.2 LTS**

# Requirement
requirements.txt目前還是手刻，若有python版本和lib版本相衝或不相容，還請自行解決。

  - **pandas == 1.2.3**
  - **keras == 2.4.3**
  - **matplotlib == 3.2.2**
  - **numpy == 1.19.5**
  - **sklearn == 0.24.1**
  - **pydot == 1.4.2**
  - **graphviz == 0.16**

# Build
Install requirement.txt
```
pip3 install -r requirements.txt
```

執行 app.py。 Input 和 Output path 已經定義在app.py里了.
```
python3 app.py
```
強烈建議直接執行ipynb檔案來直接看我們在各個區塊的輸出結果。

## Input data
Input的data為政府資料開放平臺上的[台灣電力公司_過去電力供需資訊.csv](https://data.gov.tw/dataset/19995)。在這份資料中有共有**397**個 ``entries`` 和 **71** 個 ``features``。而本模型只使用的features為 ``備轉容量 (MW)`` 與 ``備轉容量率（%)`` 。且 ``traning data`` 與 ``validation data`` 以比例為**9:1**做切割。
  <p align='center'>
    <img src='img/dataframe_head10.png'>
  </p>

  <p align='center'>
    <img src='img/train_valid.png'>
  </p>

## Scaling
為了加快模型收斂找到最佳參數組合，這裡使用``MinMaxScaler``把資料重新scaling成 **-1** 至 **1** 之間。

## Model Structure
  <p align='center'>
    <img src='img/model.png'>
  </p>
  
  <p align='center'>
    <img src='img/model_parameters.png'>
  </p>

## Training
``epochs``設定為**50**，最後``loss``約位於**0.04**左右：
 <p align='center'>
  <img src='img/loss.png'>
 </p>
 
 <p align='center'>
  <img src='img/epochs_loss.png'>
 </p>


## Prediction Result

以此模型進行2021/03/23~2021/03/29的備載容量預測結果。

  <p align='center'>
    <img src='img/result.png'>
  </p>

## Note

  1) **Features 數目*
      其實features數目不是越多越好，在這一份資料當中就含有各種各樣的features，但是有滿多都是缺值，也就是0的情況，所以蠻多都不能拿來用。我原本之前有用了前面幾個features(如: 尖峰負載，凈尖峰供電能力，水力，風力，太陽能發電等大眾特徵)，用出來的RMSE都相當高，約有400多左右，但是自從我只給模型備轉容量和備轉容量率，失誤就降下來了。

  2) **典型的Time Series 模型表現不好**
      一開始我不是就用LSTM的，我嘗試過了Naive Method, Prophet, Exponential Smoothing, SARIMA, Holt-Winter。而比較好的只有Exponential和Holt Winter 但似乎效果也沒有讓我很滿意，RMSE只有400～600之間，完全下不去。所以我個人認為還是要可以手動調整的LSTM模型的適應性會比較好，得到的結果也比較令人滿意。

  3) **Split Data的比例分配**
      偷偷將一個小秘密，如果你把train和valid的分配比例調到train越多，效果會有奇效(當然，這裡資料量少，training data越多，模型學的越好)，但是如果一般用8:2 RMSE會是400多，但是9:1 RMSE會直接掉到100多。因為平均失誤的越少，數值越低。
      
    


## Keywords
  - **Time Series**
  - **Forecasting**
  - **LSTM**
  - **RNN**
  - **Multivariables**
  - **SARIMA Model**
  - **Holt-Winter Model**

## References
  - [Dropbox Homework Description](shorturl.at/nozNX)
  - https://www.analyticsvidhya.com/blog/2020/10/multivariate-multi-step-time-series-forecasting-using-stacked-lstm-sequence-to-sequence-autoencoder-in-tensorflow-2-0-keras/
  - https://medium.datadriveninvestor.com/multivariate-time-series-using-rnn-with-keras-7f78f4488679

