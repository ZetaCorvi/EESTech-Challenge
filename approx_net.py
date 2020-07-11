import numpy as np
import torch
import pandas as pd
import datetime


data = pd.read_csv('train.csv', delimiter =';', encoding='cp1251')
data = data[data.k_power != 0]
data.dropna()
training_set = data.iloc[:, [2,3,4, 7]].values
training_set

######

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled
