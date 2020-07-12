# from google.colab import files
# uploaded = files.upload()
# from keras.models import Sequential
# from keras.layers import LSTM,Dense ,Dropout
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import csv
# #import dataset from data.csv file
# data = pd.read_csv('train.csv', delimiter =';', encoding='cp1251')
# data = data[data.k_power != 0]
# data.dropna()
# data
#
# training_set = data.iloc[:, [2,3,4]].values
# training_set
#
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range=(0,1))
# training_set_scaled = sc.fit_transform(training_set)
# training_set_scaled
#
# x_train = []
# y_train = []
# n_future = 4
# n_past = 30
# for i in range(0,len(training_set_scaled)-n_past-n_future+1):
#     x_train.append(training_set_scaled[i : i + n_past , 0])
#     y_train.append(training_set_scaled[i + n_past : i + n_past + n_future , 0 ])
# x_train , y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1) )
# regressor = Sequential()
#
#
# regressor.add(LSTM(units=600, return_sequences=True, input_shape = (x_train.shape[1],1)  ))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units= 30 , return_sequences=True))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units= 30 , return_sequences=True))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units= 60))
# regressor.add(Dropout(0.2))
# regressor.add(Dense(units = n_future,activation='linear'))
# regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
# regressor.fit(x_train, y_train, epochs=2,batch_size=32 )
#
#
#
# ###was not tested
#
# from google.colab import files
# uploaded = files.upload()
# testdataset = pd.read_csv('test.csv')
# testdataset = testdataset.iloc[:30,2:5].values
# real_temperature = pd.read_csv('test.csv')
# real_temperature = real_temperature.iloc[30:,2:5].values
# testing = sc.transform(testdataset)
# testing = np.array(testing)
# testing = np.reshape(testing,(testing.shape[1],testing.shape[0],1))
#
#

