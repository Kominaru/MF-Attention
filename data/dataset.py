from typing import Literal, Union
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch

class FastTensorDataLoader:
    """
    DataLoader-like object for a set of tensors that outspeeds
    TensorDataset + DataLoader the latter grabs individual slices and concatenates them.
    """

    def __init__(self, *tensors: torch.Tensor, batch_size: int, shuffle: bool = False):
        """
        Initialize a FastTensorDataLoader.

        Args:
            tensors (tuple of torch.Tensor): tensors to be loaded
            batch_size (int): batch size
            shuffle (bool): if True, shuffle data in-place when an iterator is created.

        Returns:
            FastTensorDataLoader: DataLoader-like object
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i : self.i + self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
            if len(self.tensors) == 1:
                batch = batch[0]
        else:
            batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
            if len(self.tensors) == 1:
                batch = batch[0]
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class DyadicRegressionDataset(Dataset):
    """
    Dataset for regression tasks on dyadic data (e.g., Collaborative Filtering).

    Attributes:
        data (pd.DataFrame): DataFrame containing the dataset with columns ['user_id', 'item_id', 'rating']

    """

    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the dataset
                Must contain the columns ['user_id', 'item_id', 'rating']
        """

        self.data = df
        self.data["user_id"] = self.data["user_id"].astype(np.int64)
        self.data["item_id"] = self.data["item_id"].astype(np.int64)
        self.data["rating"] = self.data["rating"].astype(np.float32)
        self.data = self.data[["user_id", "item_id", "rating"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = self.data.at[idx, "user_id"]
        item_id = self.data.at[idx, "item_id"]
        rating = self.data.at[idx, "rating"]

        return user_id, item_id, rating


class EmbeddingDataset(Dataset):
    """
    Dataset for embedding compression tasks
    """

    def __init__(self, embeddings: np.ndarray, ids: np.ndarray = None):
        """
        Args:
            embeddings (np.ndarray): Embeddings to be compressed
            ids (np.ndarray): Actual IDs of the embeddings. If None, IDs are assumed to be the indices of the embeddings.
        """

        self.embeddings = embeddings
        self.ids = ids if ids is not None else np.arange(len(embeddings))

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


class DyadicRegressionDataModule(LightningDataModule):
    """
    DataModule for Dyadic Regression tasks (e.g., Collaborative Filtering).

    Attributes:
        data_dir (str): Directory where the dataset is stored
        train_df (pd.DataFrame): Training dataset
        test_df (pd.DataFrame): Test dataset
        data (pd.DataFrame): Full dataset
        num_users (int): Number of users (ID of the last user + 1)
        num_items (int): Number of items (ID of the last item + 1)
        mean_rating (float): Mean rating
        min_rating (float): Minimum rating
        max_rating (float): Maximum rating
        train_dataset (DyadicRegressionDataset): Training dataset
        test_dataset (DyadicRegressionDataset): Test dataset
    """
    def __init__(
        self,
        dataset_name : str,
        split: int,
        data_dir : str = "data/datasets",
        batch_size=64,
        num_workers=4,
        verbose=False,
    ):
        """
        Creates a DyadicRegressionDataModule.

        Args:
            dataset_name (str): Name of the dataset (e.g. "ml-100k")
            split (int): Number of the split to be used
            data_dir (str): Directory where the dataset is stored
            batch_size (int): Batch size
            num_workers (int): Number of workers for the DataLoader
            verbose (bool): If True, prints basic dataset statistics
        """
        super().__init__()
        self.data_dir = data_dir 
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_df = pd.read_csv(f"{data_dir}/{dataset_name}/splits/train_{split}.csv")
        self.test_df = pd.read_csv(f"{data_dir}/{dataset_name}/splits/test_{split}.csv")
        self.data = pd.concat([self.train_df, self.test_df])

        self.num_users = self.data["user_id"].max() + 1 # ID of the last user + 1 (there may be missing IDs, but we assume they are continuous)
        self.num_items = self.data["item_id"].max() + 1
        self.mean_rating = self.data["rating"].mean()

        # Calculate the min and max ratings
        self.min_rating = self.data["rating"].min()
        self.max_rating = self.data["rating"].max()

        if verbose:
            print(f"#Users: {self.data["user_id"].nunique()} (max id {self.data['user_id'].max()})")
            print(f"#Items: {self.data["item_id"].nunique()} (max id {self.data['item_id'].max()})")
            print(f"Mean rating: {self.mean_rating:.3f}")
            print(f"Min rating: {self.min_rating:.3f}")
            print(f"Max rating: {self.max_rating:.3f}")

        self.train_df = self.train_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        self.train_dataset = DyadicRegressionDataset(self.train_df)
        self.test_dataset = DyadicRegressionDataset(self.test_df)

    def train_dataloader(self):
        return FastTensorDataLoader(
            torch.tensor(self.train_df["user_id"].values),
            torch.tensor(self.train_df["item_id"].values),
            torch.tensor(self.train_df["rating"].values),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return FastTensorDataLoader(
            torch.tensor(self.test_df["user_id"].values),
            torch.tensor(self.test_df["item_id"].values),
            torch.tensor(self.test_df["rating"].values),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader( # We use Dataloader as trainer.predict() does not support FastTensorDataLoader
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )


class EmbeddingDataModule(LightningDataModule):
    """
    Attributes:
            embeddings (np.ndarray): Embeddings to be compressed
            entity_type (Literal["user", "item"]): Type of entity ("user" or "item")
            dataset_train (EmbeddingDataset): Training dataset
            dataset_val (EmbeddingDataset): Validation dataset
            num_features (int): Number of features in the embeddings
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        data: pd.DataFrame,
        entity_type: Literal["user", "item"],
        batch_size: int = 2**10,
        num_workers: int = 0,
        val_percent: float = 0.1,
       
    ):
        """
        Creates a datamodule for embedding compression and reconstruction.

        Args:
            embeddings (np.ndarray): Embeddings to be compressed
            data (Union[pd.DataFrame, List[pd.DataFrame]]): Dataframe(s) containing the dataset reviews
            entity_type (str): Type of entity (e.g., "user", "item")
            batch_size (int): Batch size.
            num_workers (int): Number of workers for the DataLoader.
            val_percent (float): Percentage of embeddings to be used for validation.
        """
        super().__init__()
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.entity_type = entity_type
        self.num_features = embeddings.shape[1]

        if isinstance(data, pd.DataFrame):  # Obtain all the IDs (users or items) from the data
            data = [data]

        unique_ids = pd.concat([df[f"{entity_type}_id"] for df in data]).unique()
        np.random.shuffle(unique_ids)

        self.embeddings = self.embeddings[unique_ids]  # Randomize the embedding order and select only
        self.id_order = unique_ids                     # the embeddings for the unique IDs found in the data

        num_train = len(unique_ids) - int(len(unique_ids) * val_percent)

        self.dataset_train = EmbeddingDataset(self.embeddings[:num_train], unique_ids[:num_train])
        self.dataset_val = EmbeddingDataset(self.embeddings[num_train:], unique_ids[num_train:])

        print(f"Creating Embedding dataset ({len(unique_ids)} {entity_type}s) to train compression")
        print(f"  Training {entity_type}s:\t{num_train}")
        print(f"  Validation {entity_type}s:\t{len(unique_ids) - num_train}\n")

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )

        if len(self.dataset_val) > 0:
            dataloader_val = DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                shuffle=False,
            )

            return dataloader_train, dataloader_val

        return dataloader_train


class CompressorTestingCFDataModule(LightningDataModule):
    """
    DataModule for Dyadic Regression used to test an Embedding Compressor in its final CF task.

    Splits the CF validation data into reviews from users/items that will be used to train the compressor
    and reviews from users/items that will not be used to train the compressor.

    Attributes:
        user_embeddings_datamodule (EmbeddingDataModule): User Embeddings in the CF task
        item_embeddings_datamodule (EmbeddingDataModule): Item Embeddings in the CF task
        cf_val_data (pd.DataFrame): Complete Validation partition in the CF task

        df_user_t (pd.DataFrame): Reviews from users used in compressor training
        df_user_v (pd.DataFrame): Reviews from users not used in compressor training
        df_item_t (pd.DataFrame): Reviews from items used in compressor training
        df_item_v (pd.DataFrame): Reviews from items not used in compressor training
    """

    def __init__(
        self,
        user_embeddings_datamodule: EmbeddingDataModule,
        item_embeddings_datamodule: EmbeddingDataModule,
        cf_val_data: pd.DataFrame,
        batch_size: int = 2**10,
        num_workers: int = 4,
    ):
        """
        Args:
            user_embeddings_datamodule (EmbeddingDataModule): User Embeddings in the CF task
            item_embeddings_datamodule (EmbeddingDataModule): Item Embeddings in the CF task
            cf_val_data (pd.DataFrame): Validation partition in the CF task
            batch_size (int): Batch size
            num_workers (int): Number of workers for the DataLoader
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_data = cf_val_data

        print(f"Splitting CF Val Data ({len(cf_val_data)} reviews) to test compressors...")

        self.df_user_t = cf_val_data[
            cf_val_data["user_id"].isin(user_embeddings_datamodule.dataset_train.ids)
        ].reset_index(drop=True)
        self.df_user_v = cf_val_data[
            cf_val_data["user_id"].isin(user_embeddings_datamodule.dataset_val.ids)
        ].reset_index(drop=True)
        self.df_item_t = cf_val_data[
            cf_val_data["item_id"].isin(item_embeddings_datamodule.dataset_train.ids)
        ].reset_index(drop=True)
        self.df_item_v = cf_val_data[
            cf_val_data["item_id"].isin(item_embeddings_datamodule.dataset_val.ids)
        ].reset_index(drop=True)

        print(f"  from trained Users:\t{len(self.df_user_t)}")
        print(f"  from untrained Users:\t{len(self.df_user_v)}")
        print(f"  from trained Items:\t{len(self.df_item_t)}")
        print(f"  from untrained Items:\t{len(self.df_item_v)}\n")

    def val_dataloader(self, entity_type: Literal["user", "item", "both"]):
        """
        Args:
            entity_type (Literal["user", "item", "both"]) : Specifies the type how to split the validation data and return the DataLoaders.

        Returns:
            (Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader, DataLoader]]):
                - If `entity_type` is "user" or "item", returns two DataLoaders:

                    1. Reviews from users/items used in compressor training
                    2. Reviews from users/items not used in compressor training

                - If `entity_type` is "both", returns four DataLoaders:

                    1. Reviews whose users and items are used in compressor training
                    2. Reviews whose users are used in compressor training and items are not
                    3. Reviews whose items are used in compressor training and users are not
                    4. Reviews whose users and items are not used in compressor training
        """

        if entity_type in ["user", "item"]:
            return DataLoader(
                DyadicRegressionDataset(self.df_user_t if entity_type == "user" else self.df_item_t),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                shuffle=False,
            ), DataLoader(
                DyadicRegressionDataset(self.df_user_v if entity_type == "user" else self.df_item_v),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                shuffle=False,
            )
        else:
            df_both_t = self.df_user_t[self.df_user_t["item_id"].isin(self.df_item_t["item_id"])].reset_index(
                drop=True
            )
            df_both_v = self.df_user_v[self.df_user_v["item_id"].isin(self.df_item_v["item_id"])].reset_index(
                drop=True
            )
            df_user_t_item_v = self.df_user_t[self.df_user_t["item_id"].isin(self.df_item_v["item_id"])].reset_index(
                drop=True
            )
            df_user_v_item_t = self.df_user_v[self.df_user_v["item_id"].isin(self.df_item_t["item_id"])].reset_index(
                drop=True
            )

            print(f"Splitting CF Val Data ({len(self.val_data)} reviews) to test full compressors...")
            print(f"  from trained Users and Items: {len(df_both_t)}")
            print(f"  from trained Users and untrained Items: {len(df_user_t_item_v)}")
            print(f"  from untrained Users and trained Items: {len(df_user_v_item_t)}")
            print(f"  from untrained Users and Items: {len(df_both_v)}\n")

            return (
                DataLoader(
                    DyadicRegressionDataset(df_both_t),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    shuffle=False,
                ),
                DataLoader(
                    DyadicRegressionDataset(df_user_t_item_v),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    shuffle=False,
                ),
                DataLoader(
                    DyadicRegressionDataset(df_user_v_item_t),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    shuffle=False,
                ),
                DataLoader(
                    DyadicRegressionDataset(df_both_v),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    shuffle=False,
                ),
            )
