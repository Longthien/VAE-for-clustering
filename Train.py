import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, default_collate
from typing import Tuple, Callable, Optional, Union
from tqdm import tqdm
from torch.optim import Adam


def AE_train(
	dataset,
    model: torch.nn.Module,
    epochs: int,
    loss_,
    DEVICE
) -> None:
	model.train()
	dataset = dataset.to(DEVICE)

	optimizer = Adam(model.parameters(), lr=0.0005)
	print("Start training VAE...")
	for epoch in range(epochs):
	    optimizer.zero_grad()
	    x_hat, mean = model(dataset)

	    loss = loss_(dataset, x_hat, mean)/1627

	    loss.backward()
	    optimizer.step()

	    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", loss.item())
	print("Finish!!")

def  VAE_train(
	dataset,
    model: torch.nn.Module,
    epochs: int,
    loss_,
    DEVICE
) -> None:
	model.train()
	dataset = dataset.to(DEVICE)

	optimizer = Adam(model.parameters(), lr=0.0005)
	print("Start training VAE...")
	for epoch in range(epochs):
	    optimizer.zero_grad()
	    x_hat, mean, log_var = model(dataset)
	    loss = loss_(dataset, x_hat, mean, log_var)/1627

	    loss.backward()
	    optimizer.step()

	    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", loss.item())
	print("Finish!!")