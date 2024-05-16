from torch import nn
from torch.nn import functional as F


class LinkedAutoencoder(nn.Module):

    def __init__(self, d: int):
        super(LinkedAutoencoder, self).__init__()

        self.gelu = nn.GELU()

        # Down
        self.linear1 = nn.Linear(d, d // 2)
        self.bn1 = nn.BatchNorm1d(d // 2)

        self.linear2 = nn.Linear(d // 2, d // 4)
        self.bn2 = nn.BatchNorm1d(d // 4)

        self.linear3 = nn.Linear(d // 4, d // 8)

        self.linear4 = nn.Linear(d // 8, d // 4)

        self.linear5 = nn.Linear(d // 4, d // 2)

        self.linear6 = nn.Linear(d // 2, d)

    def forward(self, x):

        x1 = self.linear1(x)
        x1 = self.bn1(x1)
        x1 = self.gelu(x1)

        x2 = self.linear2(x1)
        x2 = self.bn2(x2)
        x2 = self.gelu(x2)

        x3 = self.linear3(x2)

        x4 = self.linear4(x3)
        x4 = x4 + x2
        x4 = self.gelu(x4)

        x5 = self.linear5(x4)
        x5 = x5 + x1
        x5 = self.gelu(x5)

        x6 = self.linear6(x5)

        # x6 = F.tanh(x6)

        return x6


