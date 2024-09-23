"""
Linked Autoencoder (autoencoder with residual connections) used in the embedding compression/decompression process.
"""

from torch import nn


class LinkedAutoencoder(nn.Module):
    """
    Linked Autoencoder with residual connections.

    The down part is composed by linear->batchnorm->gelu-> blocks.
    The up part is composed by linear->residual connection->gelu-> blocks.

    Fixed to have 2 hidden layers and a bottleneck of d//4.
    """

    def __init__(self, d: int):
        """
        Args:
            d (int): Input/output dimension.
        """
        super(LinkedAutoencoder, self).__init__()

        self.gelu = nn.GELU()

        # Down
        self.linear1 = nn.Linear(d, d // 2)
        self.bn1 = nn.BatchNorm1d(d // 2)

        self.linear2 = nn.Linear(d // 2, d // 4)

        # Up
        self.linear5 = nn.Linear(d // 4, d // 2)

        self.linear6 = nn.Linear(d // 2, d)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear5.weight)
        nn.init.xavier_uniform_(self.linear6.weight)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Embeddings tensor of shape (batch_size, d).
        Returns:
            torch.Tensor: Reconstructed embeddings tensor of shape (batch_size, d).
        """

        # Down
        x1 = self.linear1(x)
        x1 = self.bn1(x1)
        x1 = self.gelu(x1)

        x2 = self.linear2(x1)

        # Up
        x5 = self.linear5(x2)
        x5 += x1  # This behaves different if written as x5 = x5 + x1, reason unknown. BEWARE
        x5 = self.gelu(x5)

        x6 = self.linear6(x5)

        return x6
