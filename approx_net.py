import os

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

BATCHSIZE = 100
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
	n_layers = trial.suggest_int("n_layers", 4, 8)
	layers = []

	in_features = 8

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


def net_optimazer():
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


# optuna.visualization.plot_intermediate_values(study)


def precip_extract_data(precips: str):
	precip_statements = {
		"Явления погоды отсутствуют.": 0,
		"d": 0,
		"Без осадков.": 1,
		"Преимущественно без осадков": 2,
		"Обложной дождь": 3,
		"Ливневый дождь": 4,
		"Замерзающий (переохлажденный) дождь": 5,
		"Сильный ливневый дождь": 6,
		"Очень сильный ливневый дождь": 7,
		"Дождь со снегом": 8,
		"Сильный дождь со снегом": 9,
		"Обложной снег": 10,
		"Сильный обложной снег": 11,
		"Ливневый снег": 12,
		"Сильный ливневый снег": 13
	}

	precips = precip_statements[precips] / (len(precip_statements) - 1)
	return float(precips)


def k_power_to_float(k_power):
	"""
	Convert k_power Series: str to Series: float
	"""
	try:
		k_power = k_power.replace(",", ".")
	except:
		pass
	return float(k_power)


class AproxNet(torch.nn.Module):
	def __init__(self, n_hidden_neurons):
		super(AproxNet, self).__init__()

		self.fc1 = torch.nn.Linear(8, n_hidden_neurons)
		self.act1 = torch.nn.Sigmoid()
		self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
		self.act2 = torch.nn.Sigmoid()
		self.fc3 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
		self.act3 = torch.nn.Sigmoid()
		self.fc4 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
		self.act4 = torch.nn.Sigmoid()
		self.fc5 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
		self.act5 = torch.nn.Sigmoid()
		self.fc6 = torch.nn.Linear(n_hidden_neurons, 1)
		self.act6 = torch.nn.ReLU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.act1(x)
		x = self.fc2(x)
		x = self.act2(x)
		x = self.fc3(x)
		x = self.act3(x)
		x = self.fc4(x)
		x = self.act4(x)
		x = self.fc5(x)
		x = self.act5(x)
		x = self.fc6(x)
		x = self.act6(x)

		return (x)


def extraxt_datasets(csv_file_name: str):
	"""
	Generate formatted training and validation datasets.
		x_train,  y_validate
	"""

	dtype_dict = {"date": str,
				  "low_c": float,
				  "mid_c": float,
				  "high_c": float,
				  "cloudy": str,
				  "precip": str,
				  "k_power": str,
				  "fact": int}

	try:
		dataset = pd.read_csv(csv_file_name, encoding="cp1251", delimiter=';', dtype=dtype_dict, index_col=False)
	except FileNotFoundError:
		pass

	dataset = dataset.dropna()
	dataset = dataset.drop(columns="date").combine_first(dataset["date"].apply(date_format))
	dataset["precip"] = dataset["precip"].apply(precip_extract_data).dropna()
	dataset["k_power"] = dataset["k_power"].apply(k_power_to_float)
	dataset = dataset.drop(columns="cloudy").drop(columns="object")

	train_set = dataset.sample(frac=0.9)
	validate_set = dataset.drop(train_set.index)

	return train_set, validate_set


if __name__ == "main":
	train_data, validate_data = extraxt_datasets("datasets/train.csv")


	x_train = torch.tensor(train_data.drop(columns="fact").values.astype(float))
	y_train = torch.tensor(train_data["fact"].astype(float).values).unsqueeze_(dim=1)
	x_validate = torch.tensor(validate_data.drop(columns="fact").astype(float).values).to(DEVICE)
	y_validate = torch.tensor(validate_data["fact"].astype(float).values).to(DEVICE).unsqueeze_(dim=1)

	aprox_net = AproxNet(500).to(DEVICE)

	loss = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(aprox_net.parameters())

	batch_size = 500

	validate_loss_archive = []
	validate_accuracy_archive = []


	for epoch in range(100):

		order = np.random.permutation(len(x_train))

		loss_mean_val = 0
		accuracy_mean_val = 0

		x = None
		y = None
		for start_index in range(0, len(x_train), batch_size):
			optimizer.zero_grad()

			batch_indexes = order[start_index: start_index + batch_size]

			x_batch = x_train[batch_indexes].float().to(DEVICE)
			y_batch = y_train[batch_indexes].float().to(DEVICE)

			preds = aprox_net.forward(x_batch)
			loss_value = loss(preds, y_batch)

			loss_value.backward()
			optimizer.step()

			print(loss_value)

# validate_preds = aprox_net.forward(x_validate.float())

# validate_loss = loss(validate_preds, y_validate)
# validate_loss_archive.append(validate_loss)
# accuracy = (validate_preds.argmax(dim=1) == y_validate).float().mean()
# validate_accuracy_archive.append(accuracy)
# print(accuracy)
