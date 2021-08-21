from help_funcs import *
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import *
import torch
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    OneHotEncoder,
    LabelEncoder,
    Normalizer,
)
import pandas as pd
import wandb
from torch.optim import *
from model import Model

device = "cuda"
data = pd.read_csv("./data/data.csv")
data = torch.from_numpy(np.array(data["open_price"].iloc[::-1].tolist()))
data = data.view(1, -1)
data_input = data[:3, :-1].to(device)
data_target = data[:3, 1:].float().to(device)
model = Model()
optimizer = LBFGS(model.parameters(), lr=0.8)
criterion = MSELoss()
epochs = 100
# model = train(epochs, optimizer, data_input, model, data_target, criterion, "baseline")
pres = [
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    OneHotEncoder,
    LabelEncoder,
    Normalizer,
]
for pre in pres:
    data = pd.read_csv("./data/data.csv")
    data = data["open_price"].iloc[::-1]
    pre = pre()
    pre.fit(np.array(data).reshape(-1, 1))
    data = pre.transform(np.array(data).reshape(-1, 1))
    data = torch.from_numpy(np.array(data))
    data = data.view(1, -1)
    data_input = data[:3, :-1].to(device)
    data_target = data[:3, 1:].float().to(device)
    model = Model()
    optimizer = LBFGS(model.parameters(), lr=0.8)
    criterion = MSELoss()
    epochs = 100
    model = train(
        epochs,
        optimizer,
        data_input,
        model,
        data_target,
        criterion,
        f"{pre}-preproccessing",
    )
