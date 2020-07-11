import numpy as np
import torch
import pandas as pd
import datetime


# распиши пожалуста каждую колонку так по типам
dtype_dict = {"date": str,
			  "low_c": float,
			  "mid_c": float,
			  "high_c": float,
			  "cloudy": str,
			  "precip": str,
			  "k_power":str,
			  "fact": int}

try:
	train_set = pd.read_csv("datasets/train.csv", encoding="cp1251", delimiter=';', dtype=dtype_dict)
except FileNotFoundError:
	pass

x_train = torch.Tensor(train_set[ "low_c"].values).float()
