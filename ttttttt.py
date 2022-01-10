import numpy as np
import torch
import torch.nn as nn

x = torch.randn(53, 30, 1, 1)
shape = x.shape
net_dim = 256
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(shape[1], net_dim, kernel_size=(3,3), stride=(1,2), padding=1),
            torch.nn.Conv2d(net_dim, net_dim, kernel_size=(3,3), stride=(1,2), padding=1),
            torch.nn.Conv2d(net_dim, shape[1], kernel_size=(3,3), stride=(1,1), padding=1))
    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        return out
class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(shape[0], net_dim).nn.ReLU(),
            nn.Linear(net_dim, net_dim), nn.ReLU(),
            nn.Linear(net_dim, net_dim), nn.Hardswish(),
            torch.nn.Linear(net_dim, 5)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        return out
model = CNN()
output = model(x)
x = torch.flatten(output)
model1 = DNN()
output = model1(x)
print(x.detach().numpy().sum())

