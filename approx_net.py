import os

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

dtype_dict = {"date": str,
			  "low_c": float,
			  "mid_c": float,
			  "high_c": float,
			  "cloudy": str,
			  "precip": str,
			  "k_power": str,
			  "fact": int}

BATCHSIZE = 1000
CLASSES = 10
EPOCHS = 10
DIR = os.getcwd()
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def date_format(input_date: str):
	"""transform date string to series:{day:float , month: float, hour: float """

	date, time = input_date.split(" ")
	day, month, year = date.split(".")
	hour, _ = time.split(":")
	date_dict = {"day": float(day), "month": float(month), "hour": float(hour)}
	return pd.Series(data=date_dict, dtype=int)


def define_model(trial):
	"""
	network model for optimization by optuna
		Optimizing:
			number of layers: n_layers
		 	cells count for each layer: out_features
		 	dropout: p
	"""
	n_layers = trial.suggest_int("n_layers", 1, 6)
	layers = []

	in_features = 6

	for i in range(n_layers):
		out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
		layers.append(nn.Linear(in_features, out_features))
		layers.append(nn.ReLU())
		p = trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)
		layers.append(nn.Dropout(p))

		in_features = out_features
	layers.append(nn.Linear(in_features, 1))
	layers.append(nn.ReLU())

	return nn.Sequential(*layers)


class DatasetESS(torch.utils.data.Dataset):
	def __init__(self, csv_file):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		try:
			self.dataset = pd.read_csv(csv_file, encoding="cp1251", delimiter=';')
		except FileNotFoundError:
			pass

	def __len__(self):
		""" pass """
		return len(self.dataset)

	def __getitem__(self, idx):
		""" pass """
		if torch.is_tensor(idx):
			idx = idx.tolist()

		Y = self.dataset["fact"].iloc[idx].astype(float)
		X = self.dataset.drop(columns="fact").iloc[idx, 1:].to_numpy().astype(float)

		return X, Y


def get_dataset():
	"""load train and validate datasets"""
	train_loader = torch.utils.data.DataLoader(
		DatasetESS("train_upg.csv"),
		batch_size=BATCHSIZE,
		shuffle=True
	)
	valid_loader = torch.utils.data.DataLoader(
		DatasetESS("validate_upg.csv"),
		batch_size=BATCHSIZE,
		shuffle=True
	)

	return train_loader, valid_loader


def objective(trial):
	"""key function for optuna optimiztion"""

	model = define_model(trial).to(DEVICE)

	# Generate the optimizers.
	lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
	optimizer = optim.Adam(model.parameters(), lr=lr)

	# Get the  dataset.
	train_loader, valid_loader = get_dataset()

	# Training of the model.
	model.train()
	for epoch in range(EPOCHS):
		for batch_idx, (data, target) in enumerate(train_loader):
			# Limiting training data for faster epochs.
			if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
				break

			data, target = data.view(data.size(0), -1).to(DEVICE).float(), target.to(DEVICE).float().reshape(-1, 1)

			optimizer.zero_grad()
			output = model(data)
			loss = F.mse_loss(output, target)
			loss.backward()
			optimizer.step()

		# Validation of the model.
		model.eval()
		correct = 0
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(valid_loader):
				# Limiting validation data.
				if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
					break
				data, target = data.view(data.size(0), -1).to(DEVICE).float(), target.to(DEVICE).float().reshape(-1, 1)
				output = model(data)
				# Get the index of the max log-probability.
				pred = output.argmax(dim=1, keepdim=True)
				correct += pred.eq(target.view_as(pred)).sum().item()

		accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

		trial.report(accuracy, epoch)

		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

	return accuracy


study_name = 'study'  # Unique identifier of the study.

study = optuna.create_study(direction="maximize",
							study_name=study_name,
							storage='sqlite:///example.db',
							load_if_exists=True)

study.optimize(objective, n_trials=5, timeout=1)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
	print("    {}: {}".format(key, value))

study.trials

optuna.visualization.plot_intermediate_values(study)

try:
	dataset = pd.read_csv("train_upg.csv", encoding="cp1251", delimiter=';', dtype=dtype_dict)
except FileNotFoundError:
	pass

# dataset = dataset.combine_first(dataset["date"].apply(date_format))
# dataset = dataset.drop(columns=["cloudy", "date", "precip", "Unnamed: 0",  "object"])
# dataset.rename(columns={"fact" : "targets"})

dataset.drop("fact")


def y(x):
	try:
		x = x.replace(",", ".")
	except:
		pass
	return x


dataset['k_power'] = dataset.dropna()['k_power'].apply(y).apply(float)

# train_data = dataset.sample(frac=0.7)
# test_data = dataset.drop(train_data.index)
# train_data

ttt = torch.tensor(dataset.to_numpy().astype(float))

ttt.double()

# dataset.to_csv("train_upg.csv", sep=";", encoding="cp1251", index=False)
# dataset.to_csv("validate_upg.csv", sep=";", encoding="cp1251", index=False)

# train_data.sort_values(by="")



train_data = train_data.dropna()
test_data = test_data.dropna()

train_data[['day', 'hour', 'month', 'low_c', 'mid_c', 'hi_c', 'k_power']].values

date_format("10.11.2018 5:00")

x_train = torch.from_numpy(train_data[['day', 'hour', 'month', 'low_c', 'mid_c', 'hi_c', 'k_power']].values)

y_train = torch.from_numpy(train_data["fact"].values)
x_test = torch.from_numpy(test_data[['day', 'hour', 'month', 'low_c', 'mid_c', 'hi_c', 'k_power']].values)
y_test = torch.from_numpy(test_data["fact"].values)

x_test = x_test.to(device)
y_test = y_test.to(device)


class AproxNet(torch.nn.Module):
	def __init__(self, n_hidden_neurons):
		super(AproxNet, self).__init__()

		self.fc1 = torch.nn.Linear(7, n_hidden_neurons)
		self.act1 = torch.nn.Sigmoid()
		self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

	# self.act2 = torch.nn.Sigmoid()
	# # self.fc2 = torch.nn.LSTM(n_hidden_neurons, n_hidden_neurons)
	# self.fc3 = torch.nn.Linear(n_hidden_neurons, 1)
	# self.act3 = torch.nn.ReLU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.act1(x)
		x = self.fc2(x)
		# x = self.act2(x)
		# x = self.fc3(x)
		# x = self.act3(x)
		return (x)


aprox_net = AproxNet(500).to(device)

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(aprox_net.parameters(), lr=0.25)

x_train

y_train

!nvidia - smi

x_train.shape

y_train.shape

batch_size = 1500

test_loss_archive = []
test_accuracy_archive = []

for epoch in range(10000):

	order = np.random.permutation(len(x_train))

	loss_mean_val = 0
	accuracy_mean_val = 0

	for start_index in range(0, len(x_train), batch_size):
		optimizer.zero_grad()

		batch_indexes = order[start_index: start_index + batch_size]

		x_batch = x_train[batch_indexes].float().to(device)
		y_batch = y_train[batch_indexes].float().to(device)

		preds = aprox_net.forward(x_batch)
		loss_value = loss(preds, y_batch.reshape(-1, 1))
		loss_value.backward()
		optimizer.step()
		print(loss_value)

	# train_accuracy_archive.append(train_accuracy)
	# train_loss_archive.append(train_loss)

	test_preds = aprox_net.forward(x_test.float())

	test_loss = (test_preds.argmax(dim=1) == y_test).float().mean().item()
	test_loss_archive.append(test_loss)
	accuracy = (test_loss == y_test).float().mean()
	test_accuracy_archive.append(accuracy)

r = np.random.rand() % 500
aprox_net.forward(torch.Tensor([0, 0, 79]).to(device))

import matplotlib.pyplot as plt

plt.plot(test_loss_archive, 'b')