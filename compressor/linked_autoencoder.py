import numpy as np
from torch import nn
import torch
from torch.nn import functional as F


class LinkedAutoencoder(nn.Module):

    def __init__(self, d: int):
        super(LinkedAutoencoder, self).__init__()

        self.gelu = nn.GELU()

        # Down
        self.linear1 = nn.Linear(d, d // 2)
        self.bn1 = nn.BatchNorm1d(d // 2)

        self.linear2 = nn.Linear(d // 2, d // 4)

        self.linear5 = nn.Linear(d // 4, d // 2)

        self.linear6 = nn.Linear(d // 2, d)

        # Xaiver initialization
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear5.weight)
        nn.init.xavier_uniform_(self.linear6.weight)

    def forward(self, x):

        x1 = self.linear1(x)
        x1 = self.bn1(x1)
        x1 = self.gelu(x1)

        x2 = self.linear2(x1)
        x5 = self.linear5(x2)

        x5 += x1
        x5 = self.gelu(x5)

        x6 = self.linear6(x5)

        return x6
