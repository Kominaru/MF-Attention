from datetime import timedelta
import logging
import copy
import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from compressor.embedding_compressor import EmbeddingCompressor
from compressor.cf_validation_callback import CFValidationCallback
from mf.collaborativefiltering_mf import CollaborativeFilteringModel
from dataset import EmbeddingDataModule, DyadicRegressionDataset, CompressorTestingCFDataModule

logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(logging.WARNING)

ORIGIN_DIM = 512
TARGET_DIM = 32
DATASET = "ml-25m"
SPLIT = 5
DO_EARLY_STOPPING = False
TRAINING_TIME = "00:00:05:00"  # "DD:HH:MM:SS"


def compute_rmse(model, dataloader):
    trainer = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)

    loss = trainer.validate(model, dataloaders=dataloader, verbose=False)[0]["val_rmse"]

    return loss


def train_compressor(
    embeddings=None,
    target_dim=None,
    lr=1e-4,
    l2_reg=1e-4,
    cf_model=None,
    cf_val_datamodule=None,
):

    print(f"Training {embeddings.entity_type} compressor... (lr {lr} | l2_reg {l2_reg})")

    compressor = EmbeddingCompressor(embeddings.num_features, target_dim, lr=lr, l2_reg=l2_reg)

    # Split the CF validation data into reviews from users that will be used to train the compressor
    # and reviews from users that will not be used to train the compressor

    callbacks = []

    val_cf_callback = CFValidationCallback(
        cf_model=cf_model,
        validation_datamodule=cf_val_datamodule,
        embeddings_datamodule=embeddings,
        dataset=DATASET,
        split=SPLIT,
    )

    if os.path.exists(
        f"models/compressor/checkpoints/{DATASET}/best-model-{embeddings.num_features}-{target_dim}-{embeddings.entity_type}.ckpt"
    ):
        os.remove(
            f"models/compressor/checkpoints/{DATASET}/best-model-{embeddings.num_features}-{target_dim}-{embeddings.entity_type}.ckpt"
        )

    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath=f"models/compressor/checkpoints/{DATASET}",
        filename=f"best-model-{embeddings.num_features}-{target_dim}-{embeddings.entity_type}",
        monitor="val_loss/dataloader_idx_1" if len(cf_val_datamodule.df_user_t) > 0 else "val_loss",
        mode="min",
        train_time_interval=timedelta(minutes=5),
    )

    if DO_EARLY_STOPPING:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=("val_loss/dataloader_idx_1" if len(cf_val_datamodule.df_user_t) > 0 else "val_loss"),
                patience=5,
                mode="min",
                verbose=True,
            )
        )

    trainer = pl.Trainer(
        gpus=1,
        max_time=TRAINING_TIME,
        logger=False,
        enable_model_summary=False,
        callbacks=[val_cf_callback, checkpointer],
        num_sanity_val_steps=-1,
    )

    trainer.fit(compressor, embeddings)

    compressor = EmbeddingCompressor.load_from_checkpoint(checkpointer.best_model_path)

    predicts = trainer.predict(compressor, dataloaders=embeddings.val_dataloader())
    # Concat the predictions from various dataloaders
    if len(predicts) == 2:
        for i in range(len(predicts)):
            predicts[i] = np.concatenate(predicts[i], axis=0)
    compressed_embeddings = np.concatenate(predicts, axis=0)

    return compressed_embeddings


if __name__ == "__main__":

    print("\n" + "=" * 50)
    print(f"MF-ATTENTION Embedding Compressor")
    print(f"  Dataset: {DATASET} (split {SPLIT})")
    print(f"  Compression: {ORIGIN_DIM} -> {TARGET_DIM}")
    print(f"  Training time: {TRAINING_TIME} / compressor")
    print("=" * 50 + "\n")

    model_dir = f"models/MF/checkpoints/{DATASET}{f'/split{SPLIT}' if SPLIT else ''}"
    model_original = CollaborativeFilteringModel.load_from_checkpoint(f"{model_dir}/best-model-{ORIGIN_DIM}.ckpt")
    model_target = CollaborativeFilteringModel.load_from_checkpoint(f"{model_dir}/best-model-{TARGET_DIM}.ckpt")

    # Delete the existing compressor models if they exist
    if os.path.exists(f"models/compressor/checkpoints/{DATASET}/best-model-{ORIGIN_DIM}-{TARGET_DIM}-user.ckpt"):
        os.remove(f"models/compressor/checkpoints/{DATASET}/best-model-{ORIGIN_DIM}-{TARGET_DIM}-user.ckpt")
    if os.path.exists(f"models/compressor/checkpoints/{DATASET}/best-model-{ORIGIN_DIM}-{TARGET_DIM}-item.ckpt"):
        os.remove(f"models/compressor/checkpoints/{DATASET}/best-model-{ORIGIN_DIM}-{TARGET_DIM}-item.ckpt")

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

    user_embedding_datamodule = EmbeddingDataModule(
        user_embeddings, data=[train_data_og, test_data_og], batch_size=2**12, num_workers=4, entity_type="user"
    )

    item_embedding_datamodule = EmbeddingDataModule(
        item_embeddings, data=[train_data_og, test_data_og], batch_size=2**12, num_workers=4, entity_type="item"
    )

    cf_test_data = CompressorTestingCFDataModule(
        user_embedding_datamodule, item_embedding_datamodule, cf_val_data=test_data_og, batch_size=2**12, num_workers=4
    )

    compressed_user_embeddings = train_compressor(
        embeddings=user_embedding_datamodule,
        target_dim=TARGET_DIM,
        lr=5e-4,
        l2_reg=0,
        cf_model=copy.deepcopy(model_original),
        cf_val_datamodule=cf_test_data,
    )

    compressed_item_embeddings = train_compressor(
        embeddings=item_embedding_datamodule,
        target_dim=TARGET_DIM,
        lr=5e-4,
        l2_reg=0,
        cf_model=copy.deepcopy(model_original),
        cf_val_datamodule=cf_test_data,
    )

    trainer = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)

    losses = trainer.validate(model_original, dataloaders=cf_test_data.val_dataloader("both"))

    def print_rmse(model):
        print(f"\t All: {losses[0]['val_rmse']:.6f}")
        print(f"\t U trained, I trained: {losses[0]['val_rmse/dataloader_idx_0']:.6f}")
        print(f"\t U trained, I untrained: {losses[1]['val_rmse/dataloader_idx_1']:.6f}")
        print(f"\t U untrained, I trained: {losses[2]['val_rmse/dataloader_idx_2']:.6f}")
        print(f"\t U untrained, I untrained: {losses[3]['val_rmse/dataloader_idx_3']:.6f}")

    print(f"Original Embeddings (dim={ORIGIN_DIM})")  # Performances of the original (large) model
    print_rmse(model_original)

    print(f"Target Embeddings (dim={TARGET_DIM})")  # Performances of the target (small) model
    print_rmse(model_target)

    model_original.user_embedding.weight.data[user_embedding_datamodule.id_order] = torch.tensor(
        compressed_user_embeddings
    ).to(model_original.device)
    model_original.item_embedding.weight.data[item_embedding_datamodule.id_order] = torch.tensor(
        compressed_item_embeddings
    ).to(model_original.device)

    print(f"Compressed Embeddings (dim={TARGET_DIM})")  # Performances of the compressed (large->small) model
    print_rmse(model_original)
