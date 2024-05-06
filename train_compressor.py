import numpy as np
import pandas as pd
import torch
from model import CollaborativeFilteringModel, EmbeddingCompressor
from dataset import EmbeddingDataModule, DyadicRegressionDataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt

if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)

    train_data = pd.read_csv("compressor_data/_train.csv")
    test_data = pd.read_csv("compressor_data/_test.csv")

    model = CollaborativeFilteringModel.load_from_checkpoint(
        "models/MF/checkpoints/best-model.ckpt"
    )

    user_embeddings = model.user_embedding.weight.detach().cpu().numpy()
    item_embeddings = model.item_embedding.weight.detach().cpu().numpy()

    # Create data modules for the compressors
    user_data_module = EmbeddingDataModule(
        user_embeddings, batch_size=256, num_workers=4
    )

    item_data_module = EmbeddingDataModule(
        item_embeddings, batch_size=256, num_workers=4
    )

    # Train user compressor
    user_compressor = EmbeddingCompressor(512, 64, lr=1e-4, l2_reg=1e-4)
    trainer = pl.Trainer(
        accelerator="auto", enable_progress_bar=True, max_time="00:00:20:00"
    )

    trainer.fit(user_compressor, user_data_module)
    compressed_user_embeddings = np.concatenate(
        trainer.predict(user_compressor, dataloaders=user_data_module.test_dataloader())
    )

    item_compressor = EmbeddingCompressor(512, 64, lr=1e-4, l2_reg=1e-4)
    trainer = pl.Trainer(
        accelerator="auto", enable_progress_bar=True, max_time="00:00:20:00"
    )
    trainer.fit(item_compressor, item_data_module)
    compressed_item_embeddings = np.concatenate(
        trainer.predict(item_compressor, dataloaders=item_data_module.test_dataloader())
    )

    train_dataloader = torch.utils.data.DataLoader(
        DyadicRegressionDataset(train_data), batch_size=2**14, shuffle=False, num_workers=4, persistent_workers=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        DyadicRegressionDataset(test_data), batch_size=2**14, shuffle=False, num_workers=4, persistent_workers=True
    )

    # Compute the MSE of the obtained compressed embeddings compared to the original embeddings
    user_errors = (user_embeddings - compressed_user_embeddings).flatten()
    item_errors = (item_embeddings - compressed_item_embeddings).flatten()

    # Plot a histogram of the errors

    bins = np.linspace(0, 1, 25)*0.5

    plt.hist(abs(user_errors), bins=bins, alpha=0.5, label="User Embeddings")
    plt.hist(abs(item_errors), bins=bins, alpha=0.5, label="Item Embeddings")

    plt.yscale("log")

    plt.legend()
    plt.xlim(0, .5)
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig("compressor_data/compression_errors.pdf")
    plt.clf()
    plt.cla()

    user_mse = np.mean(abs(user_errors))
    item_mse = np.mean(abs(item_errors))

    print(f"User MSE: {user_mse}")
    print(f"Item MSE: {item_mse}")

    # Hist plot the values of the original and compressed embeddings

    bins = np.linspace(-2, 2, 50)

    plt.hist(user_embeddings.flatten(), bins=bins, alpha=0.5, label="User Embeddings")
    plt.hist(compressed_user_embeddings.flatten(), bins=bins, alpha=0.5, label="Compressed User Embeddings")

    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.xlim(-2, 2)
    plt.tight_layout()

    plt.savefig("compressor_data/user_embeddings.pdf")

    plt.clf()
    plt.cla()

    plt.hist(item_embeddings.flatten(), bins=bins, alpha=0.5, label="Item Embeddings")
    plt.hist(compressed_item_embeddings.flatten(), bins=bins, alpha=0.5, label="Compressed Item Embeddings")

    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.xlim(-2, 2)
    plt.yscale("log")
    plt.tight_layout()

    plt.savefig("compressor_data/item_embeddings.pdf")

    # Plot the maximum absolute error per each embedding

    user_max_errors = np.max(abs(user_errors).reshape(user_embeddings.shape), axis=1)
    item_max_errors = np.max(abs(item_errors).reshape(item_embeddings.shape), axis=1)

    bins = np.linspace(0, 1, 25)

    plt.hist(user_max_errors, bins=bins, alpha=0.5, label="User Embeddings")
    plt.hist(item_max_errors, bins=bins, alpha=0.5, label="Item Embeddings")

    plt.xlim(0, 1)
    plt.yscale("log")

    plt.legend()
    plt.xlabel("Max Absolute Error")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig("compressor_data/max_errors.pdf")


    #############################
    # 1. ORIGINAL EMBEDDINGS
    #############################

    train_preds = np.concatenate(
        trainer.predict(model, dataloaders=train_dataloader), axis=0
    )
    test_preds = np.concatenate(
        trainer.predict(model, dataloaders=test_dataloader), axis=0
    )

    train_rmse = np.sqrt(np.mean((train_preds - train_data["rating"].values) ** 2))
    test_rmse = np.sqrt(np.mean((test_preds - test_data["rating"].values) ** 2))

    print("Original Embeddings")
    print(f"\tTrain RMSE: {train_rmse}")
    print(f"\tTest RMSE: {test_rmse}")

    #############################
    # 2. NOISY EMBEDDINGS
    #############################

    user_errors = np.random.permutation(user_errors)
    item_errors = np.random.permutation(item_errors)

    noisy_user_embeddings = user_embeddings + user_errors.reshape(user_embeddings.shape)
    noisy_item_embeddings = item_embeddings + item_errors.reshape(item_embeddings.shape)

    model.user_embedding.weight.data = torch.tensor(noisy_user_embeddings).to(
        model.device
    )
    model.item_embedding.weight.data = torch.tensor(noisy_item_embeddings).to(
        model.device
    )

    trainer = pl.Trainer(accelerator="auto", enable_progress_bar=False)

    train_preds = np.concatenate(trainer.predict(model, train_dataloader), axis=0)
    test_preds = np.concatenate(trainer.predict(model, test_dataloader), axis=0)

    train_rmse = np.sqrt(np.mean((train_preds - train_data["rating"].values) ** 2))
    test_rmse = np.sqrt(np.mean((test_preds - test_data["rating"].values) ** 2))

    print("Noisy Embeddings")
    print(f"\tTrain RMSE: {train_rmse}")
    print(f"\tTest RMSE: {test_rmse}")

    #############################
    # 3. COMPRESSED EMBEDDINGS
    #############################

    # Check the size of the compressed embeddings is correct
    assert compressed_user_embeddings.shape == user_embeddings.shape
    assert compressed_item_embeddings.shape == item_embeddings.shape

    model.user_embedding.weight.data = torch.tensor(compressed_user_embeddings).to(
        model.device
    )
    model.item_embedding.weight.data = torch.tensor(compressed_item_embeddings).to(
        model.device
    )

    trainer = pl.Trainer(accelerator="auto", enable_progress_bar=False)
    
    train_preds = np.concatenate(trainer.predict(model, train_dataloader), axis=0)
    test_preds = np.concatenate(trainer.predict(model, test_dataloader), axis=0)

    train_rmse = np.sqrt(np.mean((train_preds - train_data["rating"].values) ** 2))
    test_rmse = np.sqrt(np.mean((test_preds - test_data["rating"].values) ** 2))

    print("Compressed Embeddings")
    print(f"\tTrain RMSE: {train_rmse}")
    print(f"\tTest RMSE: {test_rmse}")
