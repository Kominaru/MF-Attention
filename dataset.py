from typing import Literal, Union
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch

from data.data_utils import load_and_format_tripadvisor_data
from data.data_utils import load_and_format_movielens_data
from data.download_data import load_and_format_doubanmonti_data
from data.download_data import load_and_format_netflixprize_data


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
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
    Represents a dataset for regression over dyadic data.
    """

    def __init__(self, df):
        """
        Args:
            df (pandas.DataFrame): DataFrame containing the dataset
                Must contain at least the columns ['user_id', 'item_id', 'rating']
        """

        self.data = df
        self.data["user_id"] = self.data["user_id"].astype(np.int64)
        self.data["item_id"] = self.data["item_id"].astype(np.int64)
        self.data["rating"] = self.data["rating"].astype(np.float32)

    def __len__(self):
        """
        Returns:
            int: Length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: Tuple containing the user_id, item_id and rating
        """

        user_id = self.data.at[idx, "user_id"]
        item_id = self.data.at[idx, "item_id"]
        rating = self.data.at[idx, "rating"]

        return user_id, item_id, rating


class EmbeddingDataset(Dataset):
    """
    Dataset for embedding compression and reconstruction.
    """

    def __init__(self, embeddings, ids):
        """
        Args:
            embeddings (np.ndarray): Embeddings to be compressed
        """

        self.embeddings = embeddings
        self.ids = ids

    def __len__(self):
        """
        Returns:
            int: Length of the dataset
        """
        return len(self.embeddings)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            np.ndarray: Embedding
        """
        return self.embeddings[idx]


class DyadicRegressionDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=64,
        num_workers=0,
        test_size=0.1,
        dataset_name="ml-1m",
        split: int = None,
        verbose=False,
    ):
        """
        Creates a dyadic regression datamodule with a holdout train-test split.
        Downloads the dataset if it doesn't exist in the data directory.

        Args:
            data_dir (str): Directory where the dataset is stored
            batch_size (int): Batch size
            num_workers (int): Number of workers for the DataLoader
            test_size (float): Fraction of the dataset to be used as test set
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size

        if split is None:
            # Load the dataset from the file
            if dataset_name.startswith("ml-"):
                self.data = load_and_format_movielens_data(dataset_name)
            elif dataset_name.startswith("tripadvisor-"):
                self.data = load_and_format_tripadvisor_data(dataset_name)
            elif dataset_name == "netflix-prize":
                self.data = load_and_format_netflixprize_data(dataset_name)
            elif dataset_name == "douban-monti":
                self.data, self.train_df, self.test_df = load_and_format_doubanmonti_data(dataset_name)

            if dataset_name != "douban-monti":
                # Split the df into train and test sets (pandas dataframe)

                msk = np.random.rand(len(self.data)) < (1 - self.test_size)

                self.train_df = self.data[msk]
                self.test_df = self.data[~msk]

        else:
            # print(f"Using pre-split data for split {split}")
            self.train_df = pd.read_csv(f"data/{dataset_name}/splits/train_{split}.csv")
            self.test_df = pd.read_csv(f"data/{dataset_name}/splits/test_{split}.csv")
            self.data = pd.concat([self.train_df, self.test_df])

        # Calculate the number of users and items in the dataset
        self.num_users = self.data["user_id"].max() + 1
        self.num_items = self.data["item_id"].max() + 1
        self.mean_rating = self.data["rating"].mean()

        # Calculate the min and max ratings
        self.min_rating = self.data["rating"].min()
        self.max_rating = self.data["rating"].max()

        if verbose:
            print(f"#Users: {self.num_users} (max id {self.data['user_id'].max()})")
            print(f"#Items: {self.num_items} (max id {self.data['item_id'].max()})")
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
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )


class EmbeddingDataModule(LightningDataModule):
    """
    Attributes:
            embeddings (np.ndarray): Embeddings to be compressed
            batch_size (int): Batch size
            num_workers (int): Number of workers for the DataLoader
            entity_type (Literal["user", "item"]): Type of entity ("user" or "item")
            dataset_train (EmbeddingDataset): Training dataset
            dataset_val (EmbeddingDataset): Validation dataset
            num_features (int): Number of features in the embeddings
    """

    def __init__(
        self,
        embeddings: np.ndarray = None,
        data: Union[pd.DataFrame, list[pd.DataFrame]] = None,
        batch_size: int = 2**10,
        num_workers: int = 0,
        entity_type: Literal["user", "item"] = "user",
        val_percent: float = 0.1,
    ):
        """
        Creates a datamodule for embedding compression and reconstruction.

        Args:
            embeddings (np.ndarray): Embeddings to be compressed
            data (Union[pd.DataFrame, List[pd.DataFrame]]): Dataframe(s) containing the dataset reviews
            batch_size (int): Batch size
            num_workers (int): Number of workers for the DataLoader
            entity_type (str): Type of entity (e.g., "user", "item")
            val_percent (float): Percentage of embeddings to be used for validation
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
        self.id_order = unique_ids  # the embeddings for the unique IDs found in the data

        num_train = len(unique_ids) - int(len(unique_ids) * val_percent)

        self.dataset_train = EmbeddingDataset(self.embeddings[:num_train], unique_ids[:num_train])
        self.dataset_val = EmbeddingDataset(self.embeddings[num_train:], unique_ids[num_train:])

        print(f"Total embeddings: {len(self.embeddings)}")
        print(f"Train embeddings: {len(self.dataset_train)}")
        print(f"Val embeddings: {len(self.dataset_val)}")

    def train_dataloader(self):
        """
        Returns:
            torch.utils.data.DataLoader: Training DataLoader
        """
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Returns:
            torch.utils.data.DataLoader: Validation DataLoader. Returns two dataloaders if the dataset is split into train and val.
        """

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
