from torch.utils.data import Dataset
from torch import nn
import numpy as np


class riverFlowDataset(Dataset):
    def __init__(self, stations, lake):
        self.stations = stations
        self.lake = lake

    def __getitem__(self, item):
        return self.stations[item], self.lake[item]

    def __len__(self):
        return self.lake.shape[0]


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, truth, predict, bias=None):
        return self.mse_loss(truth, predict) ** 0.5



