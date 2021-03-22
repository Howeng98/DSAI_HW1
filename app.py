import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import warnings
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import math
from datetime import datetime
import csv

def calculate_rmse(valid, pred):
  rmse = 0

  print('Pred shape:',pred.shape)
  print('Valid shape:',valid.shape)
  # print(pred[0,:,0])
  # print(valid[:7,:,0])

  for index in range(y_valid.shape[0]):
    rmse += np.sqrt(np.mean((pred[index,:,0]-y_valid[index,:,0])**2))
  rmse /= y_valid.shape[0]
  print('\n===========================\n')
  print('RMSE:{}'.format(rmse))
  print('\n===========================\n')

def generate_csv(result):
  initial_date = 20210323
  try:
    with open('submission.csv', mode='w') as csv_file:
      colums = ['date','operating_reserve(MW)']
      writer = csv.DictWriter(csv_file, fieldnames=colums)
      writer.writeheader()
      for index in range(7):
        writer.writerow({'date':initial_date+index, 'operating_reserve(MW)':int(result[index])})
      csv_file.seek(0, os.SEEK_END)
      csv_file.truncate()
  except:
    raise Exception('Your result shape is not 7, please check your code!')

def split_dataset(dataset, n_past, n_future):
  X, Y = [], []
  for i in range(len(dataset)):
    if (i + n_past + n_future) > len(dataset):
      break
    x = dataset[i:i+n_past, :]
    y = dataset[i+n_past:i+n_past+n_future, :]
    X.append(x)
    Y.append(y)  
  return np.array(X), np.array(Y)





# Path
main_path = 'dataset'
print(os.listdir(main_path))


# Variables
n_past = 7
n_future = 7
n_features = 2


# Drop out the columns that we don't need
df = pd.read_csv(os.path.join(main_path,'台灣電力公司_過去電力供需資訊.csv'))
df.drop(df.iloc[:, 5:], inplace = True, axis = 1)
df.drop(df.iloc[:, 0:3], inplace = True, axis = 1)
print(df.head(10))


# Train and Valid Data Split
print(len(df))
train_df = df[:math.ceil(len(df) * 0.9)]
valid_df = df[math.ceil(len(df) * 0.9):]
print(train_df.shape)
print(valid_df.shape)


# Scaling
train = train_df
scalers = {}
for i in train_df.columns:
  scaler = MinMaxScaler(feature_range=(-1, 1))
  s = scaler.fit_transform(train[i].values.reshape(-1, 1))
  s = np.reshape(s, len(s))
  scalers['scaler_' + i] = scaler
  train[i] = s

valid = valid_df
for i in train_df.columns:
  scaler = scalers['scaler_'+i]
  s = scaler.transform(valid[i].values.reshape(-1,1))
  s = np.reshape(s, len(s))
  scalers['scaler_i'] = scaler
  valid[i] = s


# Convert Series to Samples
x_train, y_train = split_dataset(train.values, n_past, n_future)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))

x_valid, y_valid = split_dataset(valid.values, n_past, n_future)
x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], n_features))
y_valid = y_valid.reshape((y_valid.shape[0], y_valid.shape[1], n_features))

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)


# Define Model
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]
#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
#
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)
# #
model = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
model.summary()


# Compile and Fit
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
callback = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto')
model.compile(optimizer=Adam(), loss='mean_squared_error')
history = model.fit(x_train,y_train,epochs=50,validation_data=(x_valid,y_valid),batch_size=32,verbose=2,callbacks=[reduce_lr,callback])


# Ploting Model Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Valid'])
plt.show()


# Predict
pred = model.predict(x_valid)
# get date data
# df = pd.read_csv(os.path.join(main_path,'台灣電力公司_過去電力供需資訊.csv'))
# df.drop(df.iloc[:, 7:-3], inplace = True, axis = 1)
# df.drop(df.iloc[:, 0:1], inplace = True, axis = 1)
# l_time = df.to_numpy()
 

# Plotting data
# print(train.index)
# print(valid.index)
# print(pred.shape)
# cp = train.copy()
# print(cp.shape)
# cp[317:317+67,:] = pred[:,0,2]
# train['備轉容量(萬瓩)'].plot(figsize=(20, 10), fontsize=14)

# # Plotting prediction
# # print(pred.info)
# plt.subplots_adjust(left=None, bottom=None, right=3, top=2, wspace=None, hspace=None)
# # plt.subplot(2, 1, 1)
# print(pred.shape)
# xs = [datetime.strptime(d[0], '%m/%d/%Y').date() for d in l_time[2340:]]
# l1 = plt.plot(xs, pred[:,:,0])
# plt.ylim(-0.6, 0.6)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.xlabel('Time (day)')
# plt.ylabel('Operating Reserve (MW)')
# plt.grid(True)


# # Plotting validation
# # plt.subplot(2, 1, 2)

# l2 = plt.plot(xs, valid['備轉容量(MW)'][13:], '#232323')
# # valid['備轉容量(MW)'].plot(figsize=(20, 10), fontsize=14, )
# # ax = plt.gca()
# # ax.spines['bottom'].set_position(('data', 0))
# # ax.spines['left'].set_position(('data', 0))

# leg = plt.legend(l1+l2, labels=['prediction', 'validation'], loc='best')
# leg.legendHandles[0].set_color('#FF60AF')
# leg.legendHandles[1].set_color('#232323')
# plt.show()


# Denormalize
for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]    
    pred[:,:,index]=scaler.inverse_transform(pred[:,:,index])
    
    y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
    y_valid[:,:,index]=scaler.inverse_transform(y_valid[:,:,index])

# Calculate RMSE
calculate_rmse(y_valid, pred)


# Generate output result to CSV file
generate_csv(pred[0,:,0])