# DSAI_HW1   Electricity Forecasting

<p align="center">
  <img src='img/TimeSeries.jpg'>
</p>

本作業所使用的time series模型為**LSTM**，並以前七天的資料來當作每次預測的依據，去預測接下來後七天的資料。最終目標是預測20210323到20210329為期七天的備轉容量數值。評估模型的表現是以RMSE為標準，而本模型的表現結果約為 **RMSE:126.46** 。後面也會視覺化Validation的情況，我們可以根據線段的重疊度來大概判斷模型的準確率。

# Environment
  - **Python 3.8.3**
  - **Ubuntu 20.04.2 LTS**

# Requirement
requirements.txt目前還是手刻，若有python版本和lib版本相衝或不相容，還請自行解決。

  - **pandas == 1.2.3**
  - **keras == 2.4.3**
  - **matplotlib == 3.2.2**
  - **numpy == 1.19.5**

# Build
Install requirement.txt
```
pip3 install -r requirements.txt
```

Run app.py. Input and Output Path are defined in the app.py.
```
python3 app.py
```
強烈建議直接執行ipynb檔案來直接看我們在各個區塊

## Input data
作為input的資料為政府資料開放平臺上的「近三年每日尖峰備轉容量率」資料中的備轉容量率（％）與備轉容量（MW）
且testing data與validation data設定為9:1

## Scaling

## Model Structure
  <p align='center'>
    <img src='img/model.png'>
  </p>

## Training
``epochs``設定為**50**，最後``loss``約位於**0.02**左右：
<p align='center'>
  <img src='img/loss.png'>
 </p>
 

![image](https://user-images.githubusercontent.com/41318666/111903354-dfc8f900-8a7c-11eb-9f35-fbed1932b49d.png)

將所得資料視覺化後：
![image](https://user-images.githubusercontent.com/41318666/111903376-fb340400-8a7c-11eb-8c52-7e3119b605a3.png)

預測結果與validation data之對照圖：
![image](https://user-images.githubusercontent.com/41318666/111896273-5b15b500-8a53-11eb-84aa-f7486a138ffc.png)
粉線為prediction，黑線為validation

在經過計算後，可得RMSE=434左右

## Prediction Result

由於資料只有給予至2021/01/31，故往後的資料皆是基於之前所預測的結果為基礎進行預測
以此模型進行2021/03/23~2021/03/29的備載容量預測結果：

## Note

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

