import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from compressor import dds, linked_autoencoder


class EmbeddingCompressor(pl.LightningModule):
    """
    Embedding compressor based on Dynamic Data Selection (DDS) and Attention mechanisms.

    The model is composed by:
    - Encoder: LinkedAutoencoder
    - Dynamic Data Selection: DDS
    - Attention: Self attention
    - Decoder: LinkedAutoencoder

    The Encoder+DDS produces a mask that can be multiplied to the input embedding (self-attention).
    This produces a sparse-compressed embedding that is then reconstructed by the Decoder.
    """

    def __init__(
        self, d: int, features_to_select: int, lr: float = 1e-3, l2_reg: float = 0.0
    ):
        """
        Args:
            d (int): Embedding dimension.
            features_to_select (int): Number of features to select with DDS.
            lr (float): Learning rate.
            l2_reg (float): L2 regularization.
        """
        super().__init__()

        self.lr = lr
        self.l2_reg = l2_reg

        self.d = d

        self.encoder = linked_autoencoder.LinkedAutoencoder(d)
        self.dynamic_data_selection = dds.DynamicDataSelectionHard2v2(
            features_to_select
        )
        self.decoder = linked_autoencoder.LinkedAutoencoder(d)

        self.save_hyperparameters()

    def _forward(self, x):
        # Make sure the input is a tensor
        if not torch.is_tensor(x):
            try:
                x = torch.tensor(x, dtype=torch.float32)
            except:
                print(x)
        x = self.encoder(x)
        x_att, mask = self.dynamic_data_selection(x)
        x = x * x_att * mask  # Self attention
        x = self.decoder(x)

        # Compute the L2 regularization loss
        l2_reg = 0
        for param in self.encoder.parameters():
            l2_reg += torch.square(param).sum()
        for param in self.decoder.parameters():
            l2_reg += torch.square(param).sum()

        l2_reg = self.l2_reg * l2_reg / x.shape[0]

        return x, l2_reg

    def forward(self, x):
        return self._forward(x)[0]

    def training_step(self, batch, batch_idx):
        x = batch

        x_hat, l2_reg = self._forward(x)
        loss = nn.MSELoss()(x_hat, x) + l2_reg
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=True,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer
