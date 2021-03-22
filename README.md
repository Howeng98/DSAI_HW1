# DSAI_HW1   Electricity Forecasting
本作業所使用的time series模型為**LSTM**，並以前七天的資料來當作每次預測的依據，去預測接下來後七天的資料。

# Requirement
  - **pandas == 1.2.3**
  - **keras == 2.4.3**
  - **matplotlib == 3.2.2**
  - **numpy == 1.19.5**

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





