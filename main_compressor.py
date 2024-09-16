from datetime import timedelta
import logging
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from compressor.embedding_compressor import EmbeddingCompressor
from mf.collaborativefiltering_mf import CollaborativeFilteringModel
from dataset import EmbeddingDataModule, DyadicRegressionDataset
import pytorch_lightning as pl
from compressor.cf_validation_callback import CFValidationCallback
import copy

logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(logging.WARNING)

ORIGIN_DIM = 512
TARGET_DIM = 32
DATASET = "ml-25m"
SPLIT = 3
DO_EARLY_STOPPING = False
TRAINING_TIME = "00:01:00:00"  # "DD:HH:MM:SS"


def compute_rmse(model, data):
    trainer = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)

    test_dataloader = torch.utils.data.DataLoader(
        DyadicRegressionDataset(data), batch_size=2**10, shuffle=False, num_workers=4, persistent_workers=False
    )

    loss = trainer.validate(model, dataloaders=test_dataloader, verbose=False)[0]["val_rmse"]

    return loss


def train_compressor(
    embeddings=None,
    origin_dim=None,
    target_dim=None,
    lr=1e-4,
    l2_reg=1e-4,
    is_tuning=False,
    cf_model=None,
    cf_val_data=None,
    side="user",
    ids=None,
):

    data_module = EmbeddingDataModule(
        embeddings, ids=ids, batch_size=2**12 if side == "user" else 2**12, num_workers=4
    )

    compressor = EmbeddingCompressor(origin_dim, target_dim, lr=lr, l2_reg=l2_reg)

    val_percent = 0.1

    ids_known = ids[: (len(ids) - int(len(ids) * val_percent))]
    ids_unknown = ids[(len(ids) - int(len(ids) * val_percent)) :]

    print(f"Known: {len(ids_known)}, Unknown: {len(ids_unknown)}")

    cf_val_data_known = cf_val_data[cf_val_data[side + "_id"].isin(ids_known)].reset_index(drop=True)
    cf_val_data_unknown = cf_val_data[cf_val_data[side + "_id"].isin(ids_unknown)].reset_index(drop=True)

    print(f"Known: {len(cf_val_data_known)}, Unknown: {len(cf_val_data_unknown)}")

    dataloader_known = torch.utils.data.DataLoader(
        DyadicRegressionDataset(cf_val_data_known),
        batch_size=2**10,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    dataloader_unknown = torch.utils.data.DataLoader(
        DyadicRegressionDataset(cf_val_data_unknown),
        batch_size=2**10,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    callbacks = []

    val_cf_callback = CFValidationCallback(
        cf_model=cf_model,
        validation_dataloaders=(
            [dataloader_known, dataloader_unknown] if len(cf_val_data_unknown) > 0 else [dataloader_known]
        ),
        side=side,
        ids=ids,
        dataset=DATASET,
        split=SPLIT,
    )

    if os.path.exists(f"models/compressor/checkpoints/{DATASET}/best-model-{origin_dim}-{target_dim}-{side}.ckpt"):
        os.remove(f"models/compressor/checkpoints/{DATASET}/best-model-{origin_dim}-{target_dim}-{side}.ckpt")

    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath=f"models/compressor/checkpoints/{DATASET}",
        filename=f"best-model-{origin_dim}-{target_dim}-{side}",
        monitor="val_loss/dataloader_idx_1" if len(cf_val_data_unknown) > 0 else "val_loss",
        mode="min",
        train_time_interval=timedelta(minutes=5),
    )

    if DO_EARLY_STOPPING:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss/dataloader_idx_1" if len(cf_val_data_unknown) > 0 else "val_loss",
                patience=5,
                mode="min",
                verbose=True,
            )
        )

    trainer = pl.Trainer(
        gpus=1,
        enable_progress_bar=not is_tuning,
        max_time=TRAINING_TIME,
        logger=False,
        enable_model_summary=False,
        callbacks=[val_cf_callback, checkpointer],
        num_sanity_val_steps=-1,
    )

    trainer.fit(compressor, data_module)

    compressor = EmbeddingCompressor.load_from_checkpoint(checkpointer.best_model_path)

    predicts = trainer.predict(compressor, dataloaders=data_module.val_dataloader())
    # Concat the predictions from various dataloaders
    if len(predicts) == 2:
        for i in range(len(predicts)):
            predicts[i] = np.concatenate(predicts[i], axis=0)
    compressed_embeddings = np.concatenate(predicts, axis=0)

    return compressed_embeddings


if __name__ == "__main__":

    model_dir = f"models/MF/checkpoints/{DATASET}{f'/split{SPLIT}' if SPLIT else ''}"
    model_original = CollaborativeFilteringModel.load_from_checkpoint(f"{model_dir}/best-model-{ORIGIN_DIM}.ckpt")
    model_target = CollaborativeFilteringModel.load_from_checkpoint(f"{model_dir}/best-model-{TARGET_DIM}.ckpt")

    user_embeddings = model_original.user_embedding.weight.detach().cpu().numpy()
    item_embeddings = model_original.item_embedding.weight.detach().cpu().numpy()

    if SPLIT is None:

        train_data_og = pd.read_csv(f"compressor_data/{DATASET}/_train_{ORIGIN_DIM}.csv")
        test_data_og = pd.read_csv(f"compressor_data/{DATASET}/_test_{ORIGIN_DIM}.csv")

        train_data_tg = pd.read_csv(f"compressor_data/{DATASET}/_train_{TARGET_DIM}.csv")
        test_data_tg = pd.read_csv(f"compressor_data/{DATASET}/_test_{TARGET_DIM}.csv")

    else:

        train_data_og = pd.read_csv(f"data/{DATASET}/splits/train_{SPLIT}.csv")
        test_data_og = pd.read_csv(f"data/{DATASET}/splits/test_{SPLIT}.csv")

        train_data_tg = pd.read_csv(f"data/{DATASET}/splits/train_{SPLIT}.csv")
        test_data_tg = pd.read_csv(f"data/{DATASET}/splits/test_{SPLIT}.csv")

    user_ids = np.union1d(train_data_og["user_id"].unique(), test_data_og["user_id"].unique())
    item_ids = np.union1d(train_data_og["item_id"].unique(), test_data_og["item_id"].unique())

    np.random.shuffle(user_ids)
    np.random.shuffle(item_ids)

    user_embeddings = user_embeddings[user_ids]
    item_embeddings = item_embeddings[item_ids]

    compressed_user_embeddings = train_compressor(
        embeddings=user_embeddings,
        origin_dim=ORIGIN_DIM,
        target_dim=TARGET_DIM,
        lr=5e-4,
        l2_reg=0,
        cf_model=copy.deepcopy(model_original),
        cf_val_data=test_data_og,
        side="user",
        ids=user_ids,
    )
    compressed_item_embeddings = train_compressor(
        embeddings=item_embeddings,
        origin_dim=ORIGIN_DIM,
        target_dim=TARGET_DIM,
        lr=5e-4,
        l2_reg=0,
        cf_model=copy.deepcopy(model_original),
        cf_val_data=test_data_og,
        side="item",
        ids=item_ids,
    )

    val_ratio = 0.1

    known_users = user_ids[: (len(user_ids) - int(len(user_ids) * val_ratio))]
    unknown_users = user_ids[(len(user_ids) - int(len(user_ids) * val_ratio)) :]
    known_items = item_ids[: (len(item_ids) - int(len(item_ids) * val_ratio))]
    unknown_items = item_ids[(len(item_ids) - int(len(item_ids) * val_ratio)) :]

    test_data_ui = test_data_og[
        test_data_og["user_id"].isin(known_users) & test_data_og["item_id"].isin(known_items)
    ].reset_index(drop=True)

    test_data_un = test_data_og[
        test_data_og["user_id"].isin(known_users) & test_data_og["item_id"].isin(unknown_items)
    ].reset_index(drop=True)

    test_data_ni = test_data_og[
        test_data_og["user_id"].isin(unknown_users) & test_data_og["item_id"].isin(known_items)
    ].reset_index(drop=True)

    test_data_nn = test_data_og[
        test_data_og["user_id"].isin(unknown_users) & test_data_og["item_id"].isin(unknown_items)
    ].reset_index(drop=True)

    def print_rmse(model):
        print(f"\t All: {compute_rmse(model, test_data_og):.3f}")
        print(f"\t U, I known: {compute_rmse(model, test_data_ui):.3f}")
        print(f"\t U known, I unknown: {compute_rmse(model, test_data_un):.3f}")
        print(f"\t U unknown, I known: {compute_rmse(model, test_data_ni):.3f}")
        print(f"\t U, I unknown: {compute_rmse(model, test_data_nn):.3f}")

    print(f"Original Embeddings (dim={ORIGIN_DIM})")  # Performances of the original (large) model
    print_rmse(model_original)

    print(f"Target Embeddings (dim={TARGET_DIM})")  # Performances of the target (small) model
    print_rmse(model_target)

    model_original.user_embedding.weight.data[user_ids] = torch.tensor(compressed_user_embeddings).to(
        model_original.device
    )
    model_original.item_embedding.weight.data[item_ids] = torch.tensor(compressed_item_embeddings).to(
        model_original.device
    )

    print(f"Compressed Embeddings (dim={TARGET_DIM})")  # Performances of the compressed (large->small) model
    print_rmse(model_original)
