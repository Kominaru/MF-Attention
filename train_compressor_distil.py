from os import path
import os
import numpy as np
import pandas as pd
import torch
from model import CollaborativeFilteringModel, EmbeddingCompressor, CompressorCollaborativeFiltering
from dataset import EmbeddingDataModule, DyadicRegressionDataset, DyadicRegressionDataModule
import pytorch_lightning as pl
import matplotlib.pyplot as plt

if __name__ == "__main__":

    train_data = pd.read_csv("compressor_data/_train.csv") 
    test_data = pd.read_csv("compressor_data/_test.csv")

    og_model = CollaborativeFilteringModel.load_from_checkpoint(
        "models/MF/checkpoints/best-model.ckpt"
    )

    user_embeddings = og_model.user_embedding.weight
    item_embeddings = og_model.item_embedding.weight

    user_bias = og_model.user_bias.weight
    item_bias = og_model.item_bias.weight

    global_bias = og_model.global_bias
    model = CompressorCollaborativeFiltering(
        d = 512,
        features_to_select= 64,
        lr = 1e-3,
        l2_reg = 1e-4,
        user_embeddings = user_embeddings,
        item_embeddings = item_embeddings,
        user_bias = user_bias,
        item_bias = item_bias,
        global_bias = global_bias,
        rating_range=(train_data["rating"].min(), train_data["rating"].max())
    )


    print("Created model")

    train_dataloader = torch.utils.data.DataLoader(
        DyadicRegressionDataset(train_data), batch_size=2**14, shuffle=False, num_workers=4, persistent_workers=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        DyadicRegressionDataset(test_data), batch_size=2**14, shuffle=False, num_workers=4, persistent_workers=True
    )
    # Train the collaborative filtering model with compression

    print("Created Dataset")

    # Delete existing checkpoints
    if path.exists("models/MF/checkpoints/best-model-compressed.ckpt"):
        os.remove("models/MF/checkpoints/best-model-compressed.ckpt")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/MF/checkpoints",
        filename="best-model-compressed",
        monitor="val_rmse",
        mode="min",
    )

    print("Created Checkpointer")

    trainer = pl.Trainer(
        accelerator="auto",
        # logger=neptune_logger,  # Add Neptune logger
        callbacks=[checkpoint_callback],
        logger=None,
        max_time="00:05:00:00"
    )
    
    print("Created Trainer")

    trainer.fit(model, train_dataloader, test_dataloader)


    # Load the best model
    model = CompressorCollaborativeFiltering.load_from_checkpoint("models/MF/checkpoints/best-model-compressed.ckpt")

    predicts = trainer.predict(model, test_dataloader)

    predicts = np.concatenate(predicts, axis=0)

    # Compute the RMSE
    rmse = np.sqrt(np.mean((predicts - test_data["rating"].values) ** 2))
    print(f"Test RMSE: {rmse}")
