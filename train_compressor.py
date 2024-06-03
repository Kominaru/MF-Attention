import logging
import numpy as np
import pandas as pd
import torch
from model import EmbeddingCompressor
from collaborativefiltering_mf import CollaborativeFilteringModel
from dataset import EmbeddingDataModule, DyadicRegressionDataset
import pytorch_lightning as pl
from bayes_opt import BayesianOptimization

logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(logging.WARNING)

ORIGIN_DIM = 32
TARGET_DIM = 8
MODE="tune"

def compute_rmse(model, train_data, test_data):

    trainer = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)

    train_dataloader = torch.utils.data.DataLoader(
        DyadicRegressionDataset(train_data), batch_size=2**14, shuffle=False, num_workers=4, persistent_workers=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        DyadicRegressionDataset(test_data), batch_size=2**14, shuffle=False, num_workers=4, persistent_workers=True
    )

    train_preds = np.concatenate(
        trainer.predict(model, dataloaders=train_dataloader)
    )
    test_preds = np.concatenate(
        trainer.predict(model, dataloaders=test_dataloader)
    )

    train_rmse = np.sqrt(np.mean((train_preds - train_data["rating"].values) ** 2))
    test_rmse = np.sqrt(np.mean((test_preds - test_data["rating"].values) ** 2))

    print(f"\tTrain RMSE: {train_rmse}")
    print(f"\tTest RMSE: {test_rmse}")

def train_compressor(
        embeddings = None,
        origin_dim= None,
        target_dim= None,
        lr=1e-4,
        l2_reg=1e-4,
        is_tuning=False
):

    data_module = EmbeddingDataModule(embeddings, batch_size=256, num_workers=4)

    compressor = EmbeddingCompressor(origin_dim, target_dim, lr=lr, l2_reg=l2_reg)

    trainer = pl.Trainer(
        gpus=1, enable_progress_bar=not is_tuning, max_time="00:03:00:00", enable_checkpointing=False, logger=False, enable_model_summary=False
    )

    trainer.fit(compressor, data_module)

    compressed_embeddings = np.concatenate(
        trainer.predict(compressor, dataloaders=data_module.test_dataloader())
    )

    if is_tuning:
        return -compressor.last_val_loss
    else:
        return compressed_embeddings


if __name__ == "__main__":

    model_original = CollaborativeFilteringModel.load_from_checkpoint(
        f"models/MF/checkpoints/best-model-{ORIGIN_DIM}.ckpt"
    )

    model_target = CollaborativeFilteringModel.load_from_checkpoint(
        f"models/MF/checkpoints/best-model-{TARGET_DIM}.ckpt"
    )

    user_embeddings = model_original.user_embedding.weight.detach().cpu().numpy()
    item_embeddings = model_original.item_embedding.weight.detach().cpu().numpy()

    # Tune the compressor to find the best hyperparameters
    if MODE == "tune" and False:
        pbounds = {
            "l2_reg": (-6, -2),
            "lr": (-4.5, -3),
        }

        def train_compressor_tune(l2_reg, lr):
            return train_compressor(
                embeddings=user_embeddings,
                lr=10 ** lr,
                l2_reg=10 ** l2_reg,
                is_tuning=True,
            )

        optimizer = BayesianOptimization(
            f=train_compressor_tune,
            pbounds=pbounds,

        )

        optimizer.maximize(init_points=5, n_iter=5)

        print(optimizer.max)

        l2_reg = 10 ** optimizer.max["params"]["l2_reg"]
        lr = 10 ** optimizer.max["params"]["lr"]

    compressed_user_embeddings = train_compressor(user_embeddings, ORIGIN_DIM, TARGET_DIM, 1e-4, 0)

    if MODE == "tune" and False:
        pbounds = {
            "l2_reg": (-6, -2),
            "lr": (-4.5, -3),
        }

        def train_compressor_tune(l2_reg, lr):
            return train_compressor(
                embeddings=item_embeddings,
                lr=10 ** lr,
                l2_reg=10 ** l2_reg,
                is_tuning=True,
            )

        optimizer = BayesianOptimization(
            f=train_compressor_tune,
            pbounds=pbounds,

        )

        optimizer.maximize(init_points=5, n_iter=5)

        print(optimizer.max)

        l2_reg = optimizer.max["params"]["l2_reg"]
        lr = optimizer.max["params"]["lr"]

    compressed_item_embeddings = train_compressor(item_embeddings, ORIGIN_DIM, TARGET_DIM, 1e-4, 0)

    train_data_og = pd.read_csv(f"compressor_data/_train_{ORIGIN_DIM}.csv")
    test_data_og = pd.read_csv(f"compressor_data/_test_{ORIGIN_DIM}.csv")

    train_data_tg = pd.read_csv(f"compressor_data/_train_{TARGET_DIM}.csv")
    test_data_tg = pd.read_csv(f"compressor_data/_test_{TARGET_DIM}.csv")

    #############################
    # 1. ORIGINAL EMBEDDINGS
    #############################

    # Get the RSME of the original embeddings (in the original and target dims)

    
    print(f"Original Embeddings (dim={ORIGIN_DIM})")
    compute_rmse(model_original, train_data_og, test_data_og)

    print(f"Original Embeddings (dim={TARGET_DIM})")
    compute_rmse(model_target, train_data_tg, test_data_tg)

    #############################
    # 3. COMPRESSED EMBEDDINGS
    #############################

    # Check the size of the compressed embeddings is correct
    assert compressed_user_embeddings.shape == user_embeddings.shape
    assert compressed_item_embeddings.shape == item_embeddings.shape

    model_original.user_embedding.weight.data = torch.tensor(compressed_user_embeddings).to(
        model_original.device
    )
    model_original.item_embedding.weight.data = torch.tensor(compressed_item_embeddings).to(
        model_original.device
    )

    print(f"Compressed Embeddings (dim={ORIGIN_DIM} -> {TARGET_DIM})")
    compute_rmse(model_original, train_data_og, test_data_og)