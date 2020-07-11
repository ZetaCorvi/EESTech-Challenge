import numpy as np
import torch
import pandas as pd


try:
	train_set = pd.read_csv("datasets/train.csv", encoding="cp1251")
except FileNotFoundError:
	pass

x_train = torch.Tensor(train_set[])
