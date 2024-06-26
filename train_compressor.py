import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from model import EmbeddingCompressor
from collaborativefiltering_mf import CollaborativeFilteringModel
from dataset import EmbeddingDataModule, DyadicRegressionDataset, EmbeddingDataModuleSimultaneous
import pytorch_lightning as pl
from callbacks import CFValidationCallback, CFValidationCallbackSimultaneous
import copy

logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(logging.WARNING)

ORIGIN_DIM = 512
TARGET_DIM = 32
MODE="tune"
DATASET = "netflix-prize"

def compute_rmse(model, train_data, test_data, user_ids = None, item_ids = None):

    trainer = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)

    # train_dataloader = torch.utils.data.DataLoader(
    #     DyadicRegressionDataset(train_data), batch_size=2**10, shuffle=False, num_workers=1, persistent_workers=True
    # )

    if user_ids is not None:
        val_percent = 0

        known_users = user_ids[:( len(user_ids) - int(len(user_ids) * val_percent))]
        unknown_users = user_ids[( len(user_ids) - int(len(user_ids) * val_percent)):]

        known_items = item_ids[:( len(item_ids) - int(len(item_ids) * val_percent))]
        unknown_items = item_ids[( len(item_ids) - int(len(item_ids) * val_percent)):]

        test_data_known = test_data[test_data["user_id"].isin(known_users) | test_data["item_id"].isin(known_items)].reset_index(drop=True)
        test_data_unknown = test_data[test_data["user_id"].isin(unknown_users) & test_data["item_id"].isin(unknown_items)].reset_index(drop=True)

        test_dataloaders = []
        
        test_dataloaders.append(torch.utils.data.DataLoader(
            DyadicRegressionDataset(test_data_known), batch_size=2**10, shuffle=False, num_workers=1, persistent_workers=True
        ))

        if len(test_data_unknown) > 0: 
            test_dataloaders.append(

        torch.utils.data.DataLoader(
            DyadicRegressionDataset(test_data_unknown), batch_size=2**10, shuffle=False, num_workers=1, persistent_workers=True
        )
            )

    else: 
        test_dataloaders = [torch.utils.data.DataLoader(
            DyadicRegressionDataset(test_data), batch_size=2**10, shuffle=False, num_workers=1, persistent_workers=True
        )]

    # train_rmse = trainer.validate(model, dataloaders=train_dataloader, verbose=False)[0]["val_rmse"]

    test_losses = trainer.validate(model, dataloaders=test_dataloaders, verbose=False)



    # print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {test_losses[0]['val_rmse']:.3f}")

    if len(test_losses) == 2:
        val_rmse_known = test_losses[0]["val_rmse"]
        val_rmse_unknown = test_losses[1]["val_rmse"]

        print(f"Validation RMSE (known): {val_rmse_known:.4f}")
        print(f"Validation RMSE (unknown): {val_rmse_unknown:.4f}")
        

def train_compressor(
        embeddings = None,
        origin_dim= None,
        target_dim= None,
        lr=1e-4,
        l2_reg=1e-4,
        is_tuning=False,
        cf_model = None,
        cf_val_data= None,
        side = "user",
        ids = None
):

    data_module = EmbeddingDataModule(embeddings, ids=ids, batch_size=2**11 if side=='user' else 2**8, num_workers=4)

    compressor = EmbeddingCompressor(origin_dim, target_dim, lr=lr, l2_reg=l2_reg)

    val_percent = 0.1

    ids_known = ids[:( len(ids) - int(len(ids) * val_percent))]
    ids_unknown = ids[( len(ids) - int(len(ids) * val_percent)):]

    cf_val_data_known = cf_val_data[cf_val_data[side + "_id"].isin(ids_known)].reset_index(drop=True)
    cf_val_data_unknown = cf_val_data[cf_val_data[side + "_id"].isin(ids_unknown)].reset_index(drop=True)

    dataloader_known = torch.utils.data.DataLoader(
        DyadicRegressionDataset(cf_val_data_known), batch_size=2**10, shuffle=False, num_workers=4, persistent_workers=True
    )

    dataloader_unknown = torch.utils.data.DataLoader(
        DyadicRegressionDataset(cf_val_data_unknown), batch_size=2**10, shuffle=False, num_workers=4, persistent_workers=True
    )

    val_cf_callback = CFValidationCallback(
        cf_model=cf_model,
        validation_dataloaders=[dataloader_known, dataloader_unknown] if len(cf_val_data_unknown) > 0 else [dataloader_known],
        side=side,
        ids = ids,
        dataset=DATASET
    )

    trainer = pl.Trainer(
        gpus=1, enable_progress_bar=not is_tuning, max_time="00:04:00:00", enable_checkpointing=False, logger=False, enable_model_summary=False, callbacks=[val_cf_callback], num_sanity_val_steps=-1   
    )

    trainer.fit(compressor, data_module)

    predicts = trainer.predict(compressor, dataloaders=data_module.val_dataloader())
    # Concat the predictions from various dataloaders
    if len(predicts) == 2:
        for i in range(len(predicts)): 
            predicts[i] = np.concatenate(predicts[i], axis=0)
    compressed_embeddings = np.concatenate(predicts, axis=0)

    return compressed_embeddings

def train_compressor_both(
        embeddings = None,
        origin_dim= None,
        target_dim= None,
        lr=1e-4,
        l2_reg=1e-4,
        is_tuning=False,
        cf_model = None,
        cf_val_data= None,
        ids = None
):

    data_module = EmbeddingDataModuleSimultaneous(embeddings, ids=ids, batch_size=256, num_workers=4)

    compressor = EmbeddingCompressor(origin_dim, target_dim, lr=lr, l2_reg=l2_reg)

    val_cf_callback = CFValidationCallbackSimultaneous(
        cf_model=cf_model,
        validation_data=cf_val_data,
        ids = ids
    )

    trainer = pl.Trainer(
        gpus=1, enable_progress_bar=not is_tuning, max_time="00:10:00:00", enable_checkpointing=False, logger=False, enable_model_summary=False, callbacks=[val_cf_callback], num_sanity_val_steps=-1   
    )

    trainer.fit(compressor, data_module)

    predicts = trainer.predict(compressor, dataloaders=data_module.val_dataloader())
    # Concat the predictions from various dataloaders
    if len(predicts) > 1:
        for i in range(len(predicts)): 
            predicts[i] = np.concatenate(predicts[i], axis=0)
    compressed_embeddings = np.concatenate(predicts, axis=0)

    return compressed_embeddings


if __name__ == "__main__":

    model_original = CollaborativeFilteringModel.load_from_checkpoint(
        f"models/MF/checkpoints/{DATASET}/best-model-{ORIGIN_DIM}.ckpt"
    )

    model_target = CollaborativeFilteringModel.load_from_checkpoint(
        f"models/MF/checkpoints/{DATASET}/best-model-{TARGET_DIM}.ckpt"
    )

    user_embeddings = model_original.user_embedding.weight.detach().cpu().numpy()
    item_embeddings = model_original.item_embedding.weight.detach().cpu().numpy()

    train_data_og = pd.read_csv(f"compressor_data/{DATASET}/_train_{ORIGIN_DIM}.csv")
    test_data_og = pd.read_csv(f"compressor_data/{DATASET}/_test_{ORIGIN_DIM}.csv")

    train_data_tg = pd.read_csv(f"compressor_data/{DATASET}/_train_{TARGET_DIM}.csv")
    test_data_tg = pd.read_csv(f"compressor_data/{DATASET}/_test_{TARGET_DIM}.csv")

    user_ids = np.union1d(train_data_og["user_id"].unique(), test_data_og["user_id"].unique())
    item_ids = np.union1d(train_data_og["item_id"].unique(), test_data_og["item_id"].unique())

    np.random.shuffle(user_ids)
    np.random.shuffle(item_ids)

    user_embeddings = user_embeddings[user_ids]
    item_embeddings = item_embeddings[item_ids]

    # Plot the values of the embeddings
    plt.hist(user_embeddings.flatten(), bins=100, alpha=0.5, label="User Embeddings Values", color="red")
    plt.yscale("log")
    # Change to right axis
    plt.ylabel("Frequency")
    plt.xlabel("Value")
    plt.legend(loc = "upper left")


    # Change to right y-axis
    plt.twinx()
    plt.hist(item_embeddings.flatten(), bins=100, alpha=0.5, label="Item Embeddings Values", color="blue")
    plt.yscale("log")

    plt.legend(loc = "upper right")
    plt.savefig(f"compressor_data/{DATASET}/embeddings_values.pdf")
    plt.clf()

    BOTH_SIDES = False

    if BOTH_SIDES:

        compressed_embeddings = train_compressor_both((user_embeddings, item_embeddings), ORIGIN_DIM, TARGET_DIM, 1e-4, 0, cf_model=copy.deepcopy(model_original), cf_val_data=test_data_og, ids=(user_ids, item_ids))
        compressed_user_embeddings, compressed_item_embeddings  = compressed_embeddings[:len(user_ids)], compressed_embeddings[len(user_ids):]


    else:

        compressed_user_embeddings = train_compressor(user_embeddings, ORIGIN_DIM, TARGET_DIM, 1e-4, 0, cf_model=copy.deepcopy(model_original), cf_val_data=test_data_og, side="user", ids=user_ids)
        compressed_item_embeddings = train_compressor(item_embeddings, ORIGIN_DIM, TARGET_DIM, 1e-4, 0, cf_model=copy.deepcopy(model_original), cf_val_data=test_data_og, side="item", ids=item_ids)

   

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

    model_original.user_embedding.weight.data[user_ids] = torch.tensor(compressed_user_embeddings).to(
        model_original.device
    )
    model_original.item_embedding.weight.data[item_ids] = torch.tensor(compressed_item_embeddings).to(
        model_original.device
    )



    print(f"Compressed Embeddings (dim={ORIGIN_DIM} -> {TARGET_DIM})")
    compute_rmse(model_original, train_data_og, test_data_og, user_ids=user_ids, item_ids=item_ids)