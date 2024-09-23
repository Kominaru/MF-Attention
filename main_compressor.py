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
from data.dataset import (
    EmbeddingDataModule,
    CompressorTestingCFDataModule,
    DyadicRegressionDataModule,
)

logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(logging.WARNING)

ORIGIN_DIM = 512
TARGET_DIM = 32
DATASET = "ml-25m"
SPLIT = 5
DO_EARLY_STOPPING = False
TRAINING_TIME = "00:00:15:00"  # "DD:HH:MM:SS"


def train_compressor(
    embeddings=None,
    lr=1e-4,
    l2_reg=1e-4,
    cf_model=None,
    cf_val_datamodule=None,
):

    print(
        f"Training {embeddings.entity_type} compressor... (lr {lr} | l2_reg {l2_reg})"
    )

    compressor = EmbeddingCompressor(
        embeddings.num_features, TARGET_DIM, lr=lr, l2_reg=l2_reg
    )

    callbacks = []

    val_cf_callback = CFValidationCallback(
        cf_model=cf_model,
        validation_datamodule=cf_val_datamodule,
        embeddings_datamodule=embeddings,
    )

    checkpoint_dir = f"models/compressor/checkpoints/{DATASET}/{embeddings.entity_type}/best-model-{embeddings.num_features}-{TARGET_DIM}.ckpt"
    if os.path.exists(checkpoint_dir):
        os.remove(checkpoint_dir)

    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath=f"models/compressor/checkpoints/{DATASET}/{embeddings.entity_type}",
        filename=f"best-model-{embeddings.num_features}-{TARGET_DIM}",
        monitor="val_loss/dataloader_idx_1",
        mode="min",
        train_time_interval=timedelta(minutes=5),
    )

    if DO_EARLY_STOPPING:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss/dataloader_idx_1",
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
    compressed_embeddings = np.concatenate(
        [np.concatenate(p, axis=0) for p in predicts], axis=0
    )

    return compressed_embeddings


if __name__ == "__main__":

    print("\n" + "=" * 50)
    print(f"MF-ATTENTION Embedding Compressor")
    print(f"  Dataset: {DATASET} (split {SPLIT})")
    print(f"  Compression: {ORIGIN_DIM} -> {TARGET_DIM}")
    print(f"  Training time: {TRAINING_TIME} / compressor")
    print("=" * 50 + "\n")

    model_dir = f"models/MF/checkpoints/{DATASET}{f'/split{SPLIT}' if SPLIT else ''}"
    model_original = CollaborativeFilteringModel.load_from_checkpoint(
        f"{model_dir}/best-model-{ORIGIN_DIM}.ckpt"
    )
    model_target = CollaborativeFilteringModel.load_from_checkpoint(
        f"{model_dir}/best-model-{TARGET_DIM}.ckpt"
    )

    user_embeddings = model_original.user_embedding.weight.detach().cpu().numpy()
    item_embeddings = model_original.item_embedding.weight.detach().cpu().numpy()

    cf_datamodule = DyadicRegressionDataModule(
        dataset=DATASET,
        split=SPLIT,
        batch_size=2**12,
        num_workers=4,
    )

    user_embedding_datamodule = EmbeddingDataModule(
        user_embeddings=model_original.user_embedding.weight.detach().cpu().numpy(),
        data=cf_datamodule.data,
        batch_size=2**12,
        num_workers=4,
        entity_type="user",
    )
    item_embedding_datamodule = EmbeddingDataModule(
        item_embeddings=model_original.item_embedding.weight.detach().cpu().numpy(),
        data=cf_datamodule.data,
        batch_size=2**12,
        num_workers=4,
        entity_type="item",
    )

    cf_test_data = CompressorTestingCFDataModule(
        user_embeddings_datamodule=user_embedding_datamodule,
        item_embeddings_datamodule=item_embedding_datamodule,
        cf_val_data=cf_datamodule.test_df,
        batch_size=2**12,
        num_workers=4,
    )

    compressed_user_embeddings = train_compressor(
        embeddings=user_embedding_datamodule,
        lr=5e-4,
        l2_reg=0,
        cf_model=copy.deepcopy(model_original),
        cf_val_datamodule=cf_test_data,
    )

    compressed_item_embeddings = train_compressor(
        embeddings=item_embedding_datamodule,
        lr=5e-4,
        l2_reg=0,
        cf_model=copy.deepcopy(model_original),
        cf_val_datamodule=cf_test_data,
    )

    final_val_dataloaders = cf_test_data.val_dataloader("both")

    def print_rmse(model):

        trainer = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)
        losses = trainer.validate(
            model, dataloaders=final_val_dataloaders, verbose=False
        )

        print(f"\t All:\t\t\t\t{losses[0]['val_rmse']:.3f}")
        print(f"\t U trained, I trained:\t{losses[0]['val_rmse/dataloader_idx_0']:.3f}")
        print(
            f"\t U trained, I untrained:\t{losses[1]['val_rmse/dataloader_idx_1']:.3f}"
        )
        print(
            f"\t U untrained, I trained:\t{losses[2]['val_rmse/dataloader_idx_2']:.3f}"
        )
        print(
            f"\t U untrained, I untrained:\t{losses[3]['val_rmse/dataloader_idx_3']:.3f}"
        )

    print(
        f"\nOriginal Embeddings (dim={ORIGIN_DIM})"
    )  # Performances of the original (large) model
    print_rmse(model_original)

    print(
        f"\nTarget Embeddings (dim={TARGET_DIM})"
    )  # Performances of the target (small) model
    print_rmse(model_target)

    model_original.user_embedding.weight.data[user_embedding_datamodule.id_order] = (
        torch.tensor(compressed_user_embeddings).to(model_original.device)
    )
    model_original.item_embedding.weight.data[item_embedding_datamodule.id_order] = (
        torch.tensor(compressed_item_embeddings).to(model_original.device)
    )

    print(
        f"\nCompressed Embeddings (dim={TARGET_DIM})"
    )  # Performances of the compressed (large->small) model
    print_rmse(model_original)
