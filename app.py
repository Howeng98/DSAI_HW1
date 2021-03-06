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
from keras.utils import plot_model
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

# Plotting data
train_df['備轉容量(MW)'].plot(figsize=(20, 10), fontsize=14, label='train')
valid_df['備轉容量(MW)'].plot(figsize=(20, 10), fontsize=14, label='valid', title="Operating Reserve in 20200101~20210131")
# prediction = result[:,:,0]
# prediction = pd.DataFrame(prediction[-1], columns=['備轉容量(MW)'])
# prediction.plot(figsize=(20, 10), fontsize=14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Time (Day)')
plt.ylabel('Operating Reserve (MW)')
plt.legend()
plt.grid(True)
plt.show()


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

# Split dataset
x_valid, y_valid = split_dataset(valid.values, n_past, n_future)
x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], n_features))
y_valid = y_valid.reshape((y_valid.shape[0], y_valid.shape[1], n_features))

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)


# Define Model
inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
output1 = encoder_l1(inputs)
encoder_states1 = output1[1:]
encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
output2 = encoder_l2(output1[0])
encoder_states2 = output2[1:]
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(output2[0])

decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)

model = tf.keras.models.Model(inputs,decoder_outputs2)
model.summary()
plot_model(model, show_shapes=True, to_file='img/model.png')

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



# Denormalize
for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]    
    pred[:,:,index]=scaler.inverse_transform(pred[:,:,index])
    
    y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
    y_valid[:,:,index]=scaler.inverse_transform(y_valid[:,:,index])

# # Ploting
# for i in range(y_valid.shape[0]):
#   plt.plot(y_valid[i,:,0], label='Ground Truth')
#   plt.plot(pred[i,:,0], label='Prediction')
#   # valid_df['備轉容量(MW)'].plot(figsize=(20, 10), fontsize=14, label='valid', title="Operating Reserve in 20200101~20210131")
#   # prediction = result[:,:,0]
#   # prediction = pd.DataFrame(prediction[-1], columns=['備轉容量(MW)'])
#   # prediction.plot(figsize=(20, 10), fontsize=14)
#   plt.xticks(fontsize = 12)
#   plt.yticks(fontsize = 12)
#   plt.xlabel('Time (Day)')
#   plt.ylabel('Operating Reserve (MW)')
#   plt.legend()
#   plt.grid(True)
# plt.show()

# Calculate RMSE
calculate_rmse(y_valid, pred)


# Generate output result to CSV file
generate_csv(pred[0,:,0])
