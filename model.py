from typing import Literal
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from networks import LinkedAutoencoder
from dds import DynamicDataSelectionHard2, DynamicDataSelectionHard2v2

class EmbeddingCompressor(LightningModule):

    def __init__(
        self, d: int, features_to_select: int, lr: float = 1e-3, l2_reg: float = 0.0
    ):
        super().__init__()

        self.lr = lr
        self.l2_reg = l2_reg

        self.d = d

        self.encoder = LinkedAutoencoder(d)
        self.dynamic_data_selection = DynamicDataSelectionHard2v2(features_to_select)

        self.decoder = LinkedAutoencoder(d)
        self.last_val_loss = float("inf")

        self.val_outputs = []

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

        # Experiment (17/06/24): Add random gaussian noise to the input for regularization
        # x = x + torch.randn_like(x) * 0.3

        x_hat, l2_reg = self._forward(x)
        loss = nn.MSELoss()(x_hat, x) + l2_reg
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

        return loss

    def validation_step(self, batch, batch_idx , dataloader_idx=None):
        x = batch
        x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

        if (self.current_epoch < 500 and self.current_epoch % 50 == 0) or (self.current_epoch >= 500 and self.current_epoch % 250 == 0):
            self.val_outputs.append(x_hat)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        x_hat = self(x)

        return x_hat
    
    def on_validation_epoch_end(self):
        self.last_val_loss = self.trainer.callback_metrics["val_loss"].item()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer

