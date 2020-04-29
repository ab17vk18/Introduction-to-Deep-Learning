# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Step 1: Data Preprocessing

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values

#MinMax scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_training_set = scaler.fit_transform(training_set)

X_train = []
y_train = []

for i in range(60, scaled_training_set.shape[0]):
    X_train.append(scaled_training_set[i-60:i,0])
    y_train.append(scaled_training_set[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaped as required by keras.layers.recurrent and 1 describes number of
#index vectors
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

# Step 2: Building the RNN

from keras.models import Sequential
from keras.layers import (Dense,
                          LSTM,
                          Dropout)

regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, 
                   input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))  
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))  
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))  
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='Adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Step 3: Save the model

import os
curr_dir = os.path.dirname('__file__')
save_model_path = os.path.join(curr_dir,'model/simple_lstm_100ep.h5')
regressor.save(save_model_path)
print("Model saved as {}".format(save_model_path))

# Step 4: Making Predictions

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values

#Manipulating input data to generate predictions for Jan-2017
dataset_concat = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_concat[len(dataset_concat)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60,60+len(dataset_test)):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Step 5: Visualizing the results

import matplotlib.pyplot as plt

plt.plot(real_stock_price, color='red', label='Real stock price')
plt.plot(predicted_stock_price, color='blue', label='Predicted stock price')
plt.title("Google stock price trend prediction (Jan 2017)")
plt.xlabel('Day')
plt.ylabel('Price of the stock')
plt.legend()
plt.savefig('comparison.png')
plt.show()