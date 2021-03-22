# DSAI_HW1   Electricity Forecasting

<p align="center">
  <img src='img/TimeSeries.jpg'>
</p>

本作業所使用的time series模型為**LSTM**，並以前七天的資料來當作每次預測的依據，去預測接下來後七天的資料。評估模型的表現是以RMSE為標準，而該模型最後的結果是約**434.58**。後面也會視覺化validation的情況，我們可以根據線段的重疊度來大概判斷模型的準確率。

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

## Input data
作為input的資料為政府資料開放平臺上的「近三年每日尖峰備轉容量率」資料中的備轉容量率（％）與備轉容量（MW）
且testing data與validation data設定為9:1

## Model Training
epochs設定為50，每一輪所得之loss約位於0.02左右：
![image](https://user-images.githubusercontent.com/41318666/111903354-dfc8f900-8a7c-11eb-9f35-fbed1932b49d.png)

將所得資料視覺化後：
![image](https://user-images.githubusercontent.com/41318666/111903376-fb340400-8a7c-11eb-8c52-7e3119b605a3.png)

預測結果與validation data之對照圖：
![image](https://user-images.githubusercontent.com/41318666/111896273-5b15b500-8a53-11eb-84aa-f7486a138ffc.png)
粉線為prediction，黑線為validation

在經過計算後，可得RMSE=434左右

## Result

由於資料只有給予至2021/01/31，故往後的資料皆是基於之前所預測的結果為基礎進行預測
以此模型進行2021/03/23~2021/03/29的備載容量預測結果：

## Note

## Keywords

## References


