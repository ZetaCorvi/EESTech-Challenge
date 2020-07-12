import numpy as np
import torch
import pandas as pd
import datetime


# распиши пожалуста каждую колонку так по типам
dtype_dict = {"date": datetime, "low_c":float}
try:
	train_set = pd.read_csv("datasets/train.csv", encoding="cp1251", delimiter=';', dtype=dtype_dict)
except FileNotFoundError:
	pass

x_train = torch.Tensor(train_set[ "low_c"].values).float()
