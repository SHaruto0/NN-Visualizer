import torch.nn as nn

class DynamicNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation):
        super().__init__()
        layers = []
        last = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(last, h))
            layers.append(activation)
            last = h
        layers.append(nn.Linear(last, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)