import os
import numpy as np
import pytorch_lightning as pl
from os import path
from bayes_opt import BayesianOptimization
import torch
from mf.collaborativefiltering_mf import CollaborativeFilteringModel
from data.dataset import DyadicRegressionDataModule
from argparse import ArgumentParser

import logging

logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(logging.WARNING)

DATA_DIR = "data"
MODE = "train"

saved_datamodule = None

USE_BIASES = False
ACTIVATION = "sigmoid"
SIGMOID_SCALE = 1.0
EMBEDDING_DIM = 32
SPLIT = 2
DATASET = "ml-25m"

# Create command line arguments for embedding dimension, dataset, and split

args = ArgumentParser()
args.add_argument("--d", type=int, default=EMBEDDING_DIM)
args.add_argument("--dataset", type=str, default=DATASET)
args.add_argument("--split", type=int, default=SPLIT)
args = args.parse_args()

EMBEDDING_DIM = args.d
DATASET = args.dataset
SPLIT = args.split


def train_MF(
    dataset_name=DATASET,
    embedding_dim=EMBEDDING_DIM,
    data_dir="data",
    batch_size=2**15,
    num_workers=4,
    l2_reg=1e-2,
    learning_rate=5e-4,
    verbose=0,
    use_biases=USE_BIASES,
    activation=ACTIVATION,
    is_tuning=False,
    sigmoid_scale=SIGMOID_SCALE,
):

    if is_tuning:
        verbose = 0
        l2_reg = 10**l2_reg
        learning_rate = 10**learning_rate

    # Load the dyadic dataset using the data module
    data_module = (
        DyadicRegressionDataModule(
            data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            test_size=0.1,
            dataset_name=dataset_name,
            verbose=verbose,
            split=SPLIT,
        )
        if saved_datamodule is None
        else saved_datamodule
    )

    model = CollaborativeFilteringModel(
        num_users=data_module.num_users,
        num_items=data_module.num_items,
        embedding_dim=embedding_dim,
        lr=learning_rate,
        l2_reg=l2_reg,
        rating_range=(data_module.min_rating, data_module.max_rating),
        use_biases=use_biases,
        activation=activation,
        sigmoid_scale=sigmoid_scale,
    )

    if path.exists(
        f"models/MF/checkpoints/{dataset_name}/{'split' + str(SPLIT) + '/' if SPLIT is not None else ''}best-model-{EMBEDDING_DIM}.ckpt"
    ):
        os.remove(
            f"models/MF/checkpoints/{dataset_name}/{'split' + str(SPLIT) + '/' if SPLIT is not None else ''}best-model-{EMBEDDING_DIM}.ckpt"
        )

    callbacks = []

    if not is_tuning:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f"models/MF/checkpoints/{dataset_name}/{'split' + str(SPLIT) + '/' if SPLIT is not None else ''}",
            filename=f"best-model-{EMBEDDING_DIM}",
            monitor="val_rmse",
            mode="min",
        )
        callbacks.append(checkpoint_callback)

    earlystopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_rmse",
        mode="min",
        patience=5,
        verbose=False,
        min_delta=0.0001,
    )

    callbacks.append(earlystopping_callback)

    trainer = pl.Trainer(
        gpus=1,
        enable_checkpointing=not is_tuning,
        callbacks=callbacks,
        logger=False,
        precision=16,
        enable_model_summary=True,
        enable_progress_bar=True,
        max_epochs=500,
    )

    trainer.fit(model, data_module)

    if not is_tuning:

        model = CollaborativeFilteringModel.load_from_checkpoint(
            f"models/MF/checkpoints/{dataset_name}/{'split' + str(SPLIT) + '/' if SPLIT is not None else ''}best-model-{EMBEDDING_DIM}.ckpt"
        )

        predicts = trainer.predict(model, data_module.test_dataloader())
        predicts = np.concatenate(predicts, axis=0)

        rmse = np.sqrt(np.mean((predicts - data_module.test_df["rating"].values) ** 2))

        print(f"Test RMSE: {rmse:.3}")

    if not is_tuning and SPLIT is None:
        os.makedirs(f"compressor_data/{dataset_name}", exist_ok=True)
        data_module.train_df.to_csv(f"compressor_data/{dataset_name}/_train_{EMBEDDING_DIM}.csv", index=False)
        data_module.test_df.to_csv(f"compressor_data/{dataset_name}/_test_{EMBEDDING_DIM}.csv", index=False)

    if is_tuning:
        return -model.min_val_loss
    else:
        return -rmse


if __name__ == "__main__":

    if MODE == "train":
        train_MF(verbose=0)

    elif MODE == "tune":

        pbounds = {
            "l2_reg": (-6, -2),
            "learning_rate": (-4, -3),
        }

        def train_MF_tune(l2_reg, learning_rate):
            return train_MF(l2_reg=l2_reg, learning_rate=learning_rate, is_tuning=True)

        optimizer = BayesianOptimization(
            f=train_MF_tune,
            pbounds=pbounds,
        )

        optimizer.maximize(init_points=5, n_iter=10)

        print(optimizer.max)
