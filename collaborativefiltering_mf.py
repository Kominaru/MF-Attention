from typing import Literal
import torch
from torchmetrics import MeanSquaredError
from pytorch_lightning import LightningModule
from torch.nn import Embedding, Parameter

class CollaborativeFilteringModel(LightningModule):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 100,
        lr: float = 5e-4,
        l2_reg: float = 1e-5,
        rating_range: tuple = (1.0, 5.0),
        use_biases: bool = True,
        activation: Literal["logit", "sigmoid"] = "logit",
        sigmoid_scale: float = 1.0,
    ):

        super().__init__()

        self.lr = lr
        self.l2_reg = l2_reg
        self.embedding_dim = embedding_dim
        self.use_biases = use_biases
        self.min_range, self.max_range = rating_range
        self.activation = activation
        self.sigmoid_scale = sigmoid_scale

        self.user_embedding = Embedding(num_users, embedding_dim)
        self.item_embedding = Embedding(num_items, embedding_dim)

        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)

        if self.use_biases:
            self.user_bias = Embedding(num_users, 1)
            self.item_bias = Embedding(num_items, 1)
            self.global_bias = Parameter(torch.tensor(0).float())

            torch.nn.init.zeros_(self.user_bias.weight)
            torch.nn.init.zeros_(self.item_bias.weight)

        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)

        self.min_val_loss = float("inf")

        self.save_hyperparameters()
    
    def _clamp_ratings(self, x):
        return torch.clamp(x, self.min_range, self.max_range)

    def _forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        preds = torch.sum(user_embeds * item_embeds, dim=1, keepdim=True)

        if self.use_biases:
            user_bias = self.user_bias(user_ids)
            item_bias = self.item_bias(item_ids)
            preds = preds + user_bias + item_bias + self.global_bias
        
        if self.activation == "sigmoid":
            # preds = torch.sigmoid(self.sigmoid_scale * preds)
            preds = torch.sigmoid(preds) * self.sigmoid_scale - (self.sigmoid_scale - 1) / 2
            preds = self.min_range + (self.max_range - self.min_range) * preds

        l2_reg = self.l2_reg * ((torch.square(user_embeds).sum() + torch.square(item_embeds).sum()) / user_embeds.shape[0])

        return preds.squeeze(), l2_reg    
    
    def forward(self, user_ids, item_ids):
        rating_pred, _ = self._forward(user_ids, item_ids)
        return rating_pred

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings = batch
        rating_pred, l2_reg = self._forward(user_ids, item_ids)

        loss = torch.nn.MSELoss()(rating_pred, ratings) + l2_reg
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        rating_pred = self._clamp_ratings(rating_pred)
        self.train_rmse.update(rating_pred, ratings)
        self.log(
            "train_rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, ratings = batch
        rating_pred = self(user_ids, item_ids)

        loss = torch.nn.MSELoss()(rating_pred, ratings)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        rating_pred_clamped = self._clamp_ratings(rating_pred)

        self.val_rmse.update(rating_pred_clamped, ratings)
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, item_ids, _ = batch
        rating_pred = self(user_ids, item_ids)

        rating_pred_clamped = self._clamp_ratings(rating_pred)

        return rating_pred_clamped
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_rmse"].item()
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer
