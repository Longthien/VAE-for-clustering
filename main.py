from sklearnEval import evaluate
from Model import AE_model, VAE_model, Encoder, Decoder

from Train import AE_train
from LOSS import loss_function

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, default_collate
from typing import Tuple, Callable, Optional, Union
from tqdm import tqdm
from torch.optim import Adam

import pickle



def main():
	with open('X_Y.pickle', 'rb') as handle:
	    b = pickle.load(handle)
	data = torch.from_numpy(b['data'].astype('float')).float()
	target = b['label']
	kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(data)
	predict = kmeans.labels_

	DEVICE = torch.device("cpu")
	batch_size = 1
	x_dim  = 500
	hidden_dim = 300
	latent_dim = 15
	lr = 1e-4


	encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
	decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)


	ae_model = AE_model(Encoder=encoder, Decoder=decoder).to(DEVICE)
	vae_model = VAE_model(Encoder=encoder, Decoder=decoder).to(DEVICE)


	AE_train(model = ae_model, dataset = data, epochs = 20, loss_function = loss_, DEVICE = DEVICE )

	evaluate(target, predict)

if __name__ == '__main__':
	main()