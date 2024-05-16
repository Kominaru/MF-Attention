import os

# Disable pytorch lightning warnings

import numpy as np
import pandas as pd
import logging

import torch.utils
import torch.utils.data

logging.getLogger("lightning").setLevel(0)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from dataset import DyadicRegressionDataModule, EmbeddingDataModule
from model import CollaborativeFilteringModel, CrossAttentionMFModel, CollaborativeFilteringWithCompressingModel, AltCollaborativeFilteringModel
from os import path
from bayes_opt import BayesianOptimization
import neptune

MODEL = "MF"
NEPTUNE_PROJECT = "JorgePRuza-Tesis/MF-Attention"
API_KEY = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MDE0YzVjYi1hODRmLTQ4M2YtYTA0NC1mYzNjNDc5YTRlOGQifQ=="

# Needs to be in a function for PyTorch Lightning workers to work properly in Windows systems
def train_MF(
    dataset_name="ml-10m",
    embedding_dim=512,  # 128 for tripadvisor-london and ml-100k, 8 for douban-monti, 512 for the rest
    data_dir="data",
    max_epochs=1000,
    batch_size=2**15,
    num_workers=4,
    l2_reg=5e-3,  # 1e-4 for tripadvisor-london and ml-100k
    learning_rate=1e-3,  # 5e-4 for ml-100k
    dropout=0.0,
    verbose=0,
    tune=False,
):
    """
    Trains a collaborative filtering model for regression over a dyadic dataset .
    """



    # if tune:
    #     l2_reg = 10**l2_reg
    #     learning_rate = 10**learning_rate
    #     embedding_dim = int(2**embedding_dim)

    # Load the dyadic dataset using the data module
    data_module = DyadicRegressionDataModule(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        test_size=0.1,
        dataset_name=dataset_name,
        verbose=verbose,
    )

    print("data loaded")

    # Initialize the collaborative filtering model
    if MODEL == "MF":
        model = CollaborativeFilteringModel(
            data_module.num_users,
            data_module.num_items,
            embedding_dim=embedding_dim,
            l2_reg=l2_reg,
            lr=learning_rate,
            dropout=dropout,
            rating_range=(data_module.min_rating, data_module.max_rating),
        )
    elif MODEL == "CrossAttMF":
        model = CrossAttentionMFModel(
            data_module.num_users,
            data_module.num_items,
            usr_avg=data_module.avg_user_rating,
            item_avg=data_module.avg_item_rating,
            embedding_dim=embedding_dim,
            l2_reg=l2_reg,
            lr=learning_rate,
            rating_range=(data_module.min_rating, data_module.max_rating),
        )

    elif MODEL == "Compressor":
        model = CollaborativeFilteringWithCompressingModel(
            data_module.num_users,
            data_module.num_items,
            embedding_dim=embedding_dim,
            compressed_dims=64,
            l2_reg=l2_reg,
            lr=learning_rate,
            rating_range=(data_module.min_rating, data_module.max_rating),
        )

    elif MODEL == "AltMF":

        model = AltCollaborativeFilteringModel(
            data_module.num_users,
            data_module.num_items,
            embedding_dim=embedding_dim,
            l2_reg=l2_reg,
            lr=learning_rate,
            rating_range=(data_module.min_rating, data_module.max_rating),
        )

    # # Checkpoint only the weights that give the best validation RMSE, overwriting existing checkpoints
    if path.exists("models/MF/checkpoints/best-model.ckpt"):
        os.remove("models/MF/checkpoints/best-model.ckpt")

    # Add Neptune logger
    # neptune_logger = pl.loggers.NeptuneLogger(
    #     api_key=API_KEY,
    #     project=NEPTUNE_PROJECT,
    #     log_model_checkpoints=False,
    # )

    # neptune_logger.experiment["parameters/embedding_dim"] = embedding_dim
    # neptune_logger.experiment["parameters/l2_reg"] = l2_reg
    # neptune_logger.experiment["parameters/learning_rate"] = learning_rate
    # neptune_logger.experiment["parameters/model"] = MODEL
    # neptune_logger.experiment["parameters/dataset"] = dataset_name

    # Checkpoint to local the best model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/MF/checkpoints",
        filename="best-model",
        monitor="val_rmse",
        mode="min",
    )

    # # Initialize the trainer
    trainer = pl.Trainer(
        accelerator="auto",
        # logger=neptune_logger,  # Add Neptune logger
        callbacks=[checkpoint_callback],
        enable_model_summary=verbose,
        enable_progress_bar=verbose,
        max_time="00:00:10:00"
    )

    # print("training")

    # # Train the model
    trainer.fit(model, data_module)

    # Load the best model
    if MODEL == "MF":
        model = CollaborativeFilteringModel.load_from_checkpoint("models/MF/checkpoints/best-model.ckpt")
    elif MODEL == "CrossAttMF":
        model = CrossAttentionMFModel.load_from_checkpoint("models/MF/checkpoints/best-model.ckpt")
    elif MODEL == "Compressor":
        model = CollaborativeFilteringWithCompressingModel.load_from_checkpoint("models/MF/checkpoints/best-model.ckpt")
    elif MODEL == "AltMF":
        model = AltCollaborativeFilteringModel.load_from_checkpoint("models/MF/checkpoints/best-model.ckpt")    

    predicts = trainer.predict(model, data_module.test_dataloader())

    predicts = np.concatenate(predicts, axis=0)

    # Compute the RMSE
    rmse = np.sqrt(np.mean((predicts - data_module.test_df["rating"].values) ** 2))
    print(f"Test RMSE: {rmse}")

    # Save the train and test partitions as CSV files

    os.makedirs("compressor_data", exist_ok=True)

    data_module.train_df.to_csv(f"compressor_data/_train.csv", index=False)
    data_module.test_df.to_csv(f"compressor_data/_test.csv", index=False)    


if __name__ == "__main__":
    MODE = "train"

    if MODE == "train":
        train_MF(verbose=1)

    elif MODE == "tune":

        # Bayesian optimization

        # Bounded region of parameter space
        pbounds = {
            "embedding_dim": (2, 6),  # 8 to 1024
            "l2_reg": (-6, -2),  # 1e-6 to 1e-2
            "learning_rate": (-5, -3),  # 1e-5 to 1e-2
        }

        def train_MF_tune(embedding_dim, l2_reg, learning_rate):
            return train_MF(
                embedding_dim=embedding_dim, l2_reg=l2_reg, learning_rate=learning_rate, tune=True, save_outputs=False
            )

        optimizer = BayesianOptimization(
            f=train_MF_tune,
            pbounds=pbounds,
        )

        optimizer.maximize(init_points=10, n_iter=30)

        print(optimizer.max)
