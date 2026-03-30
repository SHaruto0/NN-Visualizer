import torch.nn as nn

activations_dict = {
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "Sigmoid": nn.Sigmoid(),
    "LeakyReLU": nn.LeakyReLU(),
    "ELU": nn.ELU(),
    "GELU": nn.GELU()
}

losses_dict = {
    "MSE": nn.MSELoss(),
    "MAE": nn.L1Loss(),
    "Huber": nn.SmoothL1Loss()
}