from torch.optim import Adam
import torch.nn as nn
import torch


def AE_loss_function(x, x_hat, mean):
    loss_ = nn.MSELoss(reduction='mean')
    loss = loss_(x_hat, x)
    
    return loss 

def VAE_loss_function(x, x_hat, mean, log_var):
    loss_ = nn.MSELoss(reduction='mean')
    loss = loss_(x_hat, x)
    KLD  = - 0.5 * torch.mean(1+ log_var - mean.pow(2) - log_var.exp())

    return loss + KLD

