from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import numpy as np
from dds import DynamicDataSelectionHard2
from networks import LinkedAutoencoder


class CollaborativeFilteringModel(LightningModule):
    """
    Collaborative filtering model that predicts the rating of a item for a user
    Ratings are computed as:
        r_hat = dot(user_embedding, item_embedding) + user_bias + item_bias + global_bias

    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 100,
        lr: float = 5e-4,
        l2_reg: float = 1e-5,
        dropout: float = 0.0,
        rating_range: tuple = (1.0, 5.0),
    ):
        """
        Initializes a Collaborative Filtering model for item ratings prediction

        Args:
            num_users (int): Number of users in the dataset
            num_items (int): Number of items in the dataset
            embedding_dim (int): Embedding size for user and item
            lr (float): Learning rate
            l2_reg (float): L2 regularization coefficient

        """
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0).float())
        self.min_range, self.max_range = rating_range

        self.user_dropout = nn.Dropout(dropout)
        self.item_dropout = nn.Dropout(dropout)

        self.lr = lr
        self.l2_reg = l2_reg

        self.loss_fn = nn.MSELoss()

        # Initialize embedding weights with Xavier distribution
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Initialize bias weights to zero
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)

        self.clamp_ratings = lambda x: torch.clamp(x, min=rating_range[0], max=rating_range[1])

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        user_embeds = self.user_dropout(user_embeds)
        item_embeds = self.item_dropout(item_embeds)

        dot_product = torch.sum(user_embeds * item_embeds, dim=1, keepdim=True)

        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(item_ids)

        prediction = (
            dot_product + user_bias + item_bias + self.global_bias
        )  # We add the user and item biases and the global bias

        return prediction

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, ratings = batch
        rating_pred = self(user_ids, item_ids).squeeze()

        # Compute Mean Squared Error (MSE) loss
        # Notice we do not clamp the ratings
        loss = nn.MSELoss()(rating_pred, ratings)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        rating_pred = self.clamp_ratings(rating_pred)
        self.train_rmse.update(rating_pred, ratings)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, ratings = batch
        rating_pred = self(user_ids, item_ids).squeeze()

        loss = nn.MSELoss()(rating_pred, ratings)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Clamp the predicted ratings for RMSE computation
        rating_pred_clamped = self.clamp_ratings(rating_pred)

        self.val_rmse.update(rating_pred_clamped, ratings)
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, item_ids, _ = batch
        rating_pred = self(user_ids, item_ids).squeeze()

        # Clamp the predicted ratings between 1.0 and 5.0
        rating_pred_clamped = self.clamp_ratings(rating_pred)

        return rating_pred_clamped

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)

        return optimizer


def init_weights(m):
    if not isinstance(m, nn.BatchNorm2d):
        if hasattr(m, "weight"):
            torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0.001)

class CrossAttentionMFModel(LightningModule):
    """
    Model that predicts the rating of a item for a user using a cross-attention mechanism
    Ratings are computed as:
        item_att = softmax(embed_u_avg(user_avg)) * embed_i(item)
        user_att = softmax(embed_i_avg(item_avg)) * embed_u(user)
        r_hat = dot(item_att, user_att)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        usr_avg: np.ndarray,
        item_avg: np.ndarray,
        embedding_dim: int = 100,
        lr: float = 5e-4,
        l2_reg: float = 0.,
        rating_range: tuple = (1.0, 5.0),
    ):
        """
        Initializes a Collaborative Filtering model for item ratings prediction

        Args:
            num_users (int): Number of users in the dataset
            num_items (int): Number of items in the dataset
            usr_avg (int): Average rating of the users
            item_avg (int): Average rating of the items
            embedding_dim (int): Embedding size for user and item
            lr (float): Learning rate
            l2_reg (float): L2 regularization coefficient

        """
        super().__init__()

        embedding_dim = int(embedding_dim)

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0).float())

        # dxd attention masks
        self.user_att = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.SELU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 2),
            nn.SELU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )

        self.item_att = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.SELU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 2),
            nn.SELU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )

        self.user_avg = torch.Tensor(usr_avg).to("cuda")
        self.item_avg = torch.Tensor(item_avg).to("cuda")

        self.lr = lr
        self.l2_reg = l2_reg

        self.loss_fn = nn.MSELoss()

        self.min_rating, self.max_rating = rating_range

        self.dds = DynamicDataSelectionHard2(n_features_to_select=.05)

        self.apply(init_weights)

        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)

        self.clamp_ratings = lambda x: torch.clamp(x, min=rating_range[0], max=rating_range[1])

        # Save hyperparameters
        self.save_hyperparameters()



    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)  # User embed (d x 1)
        item_embeds = self.item_embedding(item_ids)  # Item embed (d x 1)

        user_avgs = self.user_avg[user_ids].unsqueeze(1)  # User average rating (d x 1)
        item_avgs = self.item_avg[item_ids].unsqueeze(1)  # Item average rating (d x 1)

        # Attention masks (d x d)
        user_atts = self.user_att(user_avgs) # .reshape(-1, user_embeds.shape[1], user_embeds.shape[1])
        item_atts = self.item_att(item_avgs) # .reshape(-1, item_embeds.shape[1], item_embeds.shape[1])

        # print("Attention mask: ", user_atts.shape)

        mask_u, user_atts = self.dds(user_atts)  # User attention mask (d x d)
        mask_i, item_atts = self.dds(item_atts)  # Item attention mask (d x d)
        # Cross attention

        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(item_ids)

        preds = torch.sum(user_embeds * item_embeds * (item_atts * mask_i + mask_u * user_atts) / 2., dim=-1, keepdim=True)
        preds = preds + user_bias + item_bias + self.global_bias

        return preds.transpose(0, 1)

    def training_step(self, batch):
        user_ids, item_ids, ratings = batch
        rating_pred = self(user_ids, item_ids).squeeze()

        loss = nn.MSELoss()(rating_pred, ratings)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        rating_pred = self.clamp_ratings(rating_pred)
        self.train_rmse.update(rating_pred, ratings)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, ratings = batch
        rating_pred = self(user_ids, item_ids).squeeze()

        loss = nn.MSELoss()(rating_pred, ratings)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        rating_pred= self.clamp_ratings(rating_pred)
        self.val_rmse.update(rating_pred, ratings)
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        user_ids, item_ids, _ = batch
        rating_pred = self(user_ids, item_ids).squeeze()

        # Clamp the predicted ratings between 1.0 and 5.0
        rating_pred_clamped = self.clamp_ratings(rating_pred)

        return rating_pred_clamped

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)

        return optimizer
    
class EmbeddingCompressor(LightningModule):
    
        def __init__(self, d: int, features_to_select: int, lr: float = 1e-3, l2_reg: float = 0.):
            super().__init__()

            self.lr = lr
            self.l2_reg = l2_reg

            self.d = d
    
            self.encoder = LinkedAutoencoder(d)
            self.dynamic_data_selection = DynamicDataSelectionHard2(features_to_select)

            self.decoder = LinkedAutoencoder(d)

        def forward(self, x):
            # Make sure the input is a tensor
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            x = self.encoder(x)
            x_att, mask = self.dynamic_data_selection(x)
            x = x * x_att * mask # Self attention
            x = self.decoder(x)

            return x
        
        def training_step(self, batch, batch_idx):
            x = batch
            x_hat = self(x)
            loss = nn.MSELoss()(x_hat, x)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

            return loss
        
        def validation_step(self, batch, batch_idx):
            x = batch
            x_hat = self(x)
            loss = nn.MSELoss()(x_hat, x)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        
        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            x = batch
            x_hat = self(x)

            return x_hat

        def configure_optimizers(self):
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)

            return optimizer