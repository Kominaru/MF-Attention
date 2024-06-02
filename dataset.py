import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import h5py
import torch

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
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
            if len(self.tensors) == 1:
                batch = batch[0]
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
            if len(self.tensors) == 1:
                batch = batch[0]
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def load_and_format_tripadvisor_data(dataset_name):
    """
    Formats a raw TripAdvisor dataset into a the standard user/item format used in this project.
    The datasets are available at https://zenodo.org/record/5644892#.YmZ3Z2gzZPY

    Args:
        dataset_name (str): Name of the dataset (e.g. tripadvisor-paris)

    Returns:
        pandas.DataFrame: DataFrame containing the formatted dataset with the columns ['user_id', 'item_id', 'rating']
    """

    df = pd.read_pickle(os.path.join("data", dataset_name, "reviews.pkl"))
    df = df[["userId", "restaurantId", "rating"]]
    df.columns = ["user_id", "item_id", "rating"]

    # Drop items with less than 100 ratings and users with less than 20 ratings
    # Remove repeated user-item pairs
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    df = df.groupby("item_id").filter(lambda x: len(x) >= 10)
    df = df.groupby("user_id").filter(lambda x: len(x) >= 10)

    # Remove NA values
    df = df.dropna()

    # Create new user and item ids ( userId's are strings, restaurantId's are not continuous)
    df["user_id"] = df["user_id"].astype("category").cat.codes
    df["item_id"] = df["item_id"].astype("category").cat.codes

    df["rating"] = df["rating"] / 10

    return df


def load_and_format_movielens_data(dataset_name):
    """
    Formats a raw MovieLens dataset into a the standard user/item format used in this project.

    Args:
        dataset_name (str): Name of the dataset (e.g. ml-1m)

    Returns:
        pandas.DataFrame: DataFrame containing the formatted dataset with the columns ['user_id', 'item_id', 'rating']
    """
    if dataset_name == "ml-100k":
        df = pd.read_csv(os.path.join("data", dataset_name, "u.data"), sep="\t", header=None)
    elif dataset_name in ["ml-1m", "ml-10m"]:
        df = pd.read_csv(os.path.join("data", dataset_name, "ratings.dat"), sep="::", engine="python", header=None)
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    df = df[["user_id", "item_id", "rating"]]

    return df


def load_and_format_doubanmonti_data(dataset_name):
    # Open douban-monti dataset (matlab file) using h5py
    with h5py.File(os.path.join("data", dataset_name, "training_test_dataset.mat"), "r") as f:
        # Convert to numpy arrays

        data = np.array(f["M"])
        train_data = np.array(f["M"]) * np.array(f["Otraining"])
        test_data = np.array(f["M"]) * np.array(f["Otest"])

    def rating_matrix_to_dataframe(ratings: np.ndarray):
        """
        Converts a rating matrix to a pandas DataFrame.

        Args:
            ratings (np.ndarray): Rating matrix

        Returns:
            pandas.DataFrame: DataFrame containing the ratings
        """

        # Get the indices of the non-zero ratings
        nonzero_indices = np.nonzero(ratings)

        # Create the dataframe
        df = pd.DataFrame(
            {
                "user_id": nonzero_indices[0],
                "item_id": nonzero_indices[1],
                "rating": ratings[nonzero_indices],
            }
        )

        # Min and max ratings
        min_rating = df["rating"].min()
        max_rating = df["rating"].max()

        return df

    # Convert the training and test data to dataframes
    all_df = rating_matrix_to_dataframe(data)
    train_df = rating_matrix_to_dataframe(train_data)
    test_df = rating_matrix_to_dataframe(test_data)

    return all_df, train_df, test_df


def load_and_format_netflixprize_data(dataset_name):
    """
    Formats a raw Netflix Prize dataset into a the standard user/item format used in this project.
    The datasets are available at https://www.kaggle.com/netflix-inc/netflix-prize-data

    Args:
        dataset_name (str): Name of the dataset (netflix-prize)

    Returns:
        pandas.DataFrame: DataFrame containing the formatted dataset with the columns ['user_id', 'item_id', 'rating']
    """

    # Load the data
    df = pd.read_csv("data/netflix-prize/ratings.csv")

    # Rename the columns
    df.columns = ["user_id", "item_id", "rating"]

    # Make the ids start from 0 by creating new user and item ids
    df["user_id"] = df["user_id"].astype("category").cat.codes
    df["item_id"] = df["item_id"].astype("category").cat.codes

    # Convert ratings to float
    df["rating"] = df["rating"].astype(np.float32)

    return df


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

    def __init__(self, embeddings):
        """
        Args:
            embeddings (np.ndarray): Embeddings to be compressed
        """

        self.embeddings = embeddings

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
    def __init__(self, data_dir, batch_size=64, num_workers=0, test_size=0.1, dataset_name="ml-1m", verbose=True):
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
    def __init__(self, embeddings, batch_size=64, num_workers=0):
        """
        Creates a datamodule for embedding compression and reconstruction.

        Args:
            embeddings (np.ndarray): Embeddings to be compressed
            batch_size (int): Batch size
            num_workers (int): Number of workers for the DataLoader
        """
        super().__init__()
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = EmbeddingDataset(self.embeddings)

    def train_dataloader(self):
        """
        Returns:
            torch.utils.data.DataLoader: DataLoader for the training set
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Returns:
            torch.utils.data.DataLoader: DataLoader for the validation set (same as test set)
        """
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )
    
    def test_dataloader(self):
        """
        Returns:
            torch.utils.data.DataLoader: DataLoader for the test set
        """
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=False
        )
    

class DyadicRegressionDistilDataset(Dataset):
    """
    Represents a dataset for regression over dyadic data.
    """

    def __init__(self, df, user_embeds, item_embeds, user_bias, item_bias, global_bias):

        self.data = df
        self.data["user_id"] = self.data["user_id"].astype(np.int64)
        self.data["item_id"] = self.data["item_id"].astype(np.int64)
        self.data["rating"] = self.data["rating"].astype(np.float32)

        self.user_embeds = user_embeds
        self.item_embeds = item_embeds
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.global_bias = global_bias

    def __len__(self):
        """
        Returns:
            int: Length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        
        user_id = self.data.at[idx, "user_id"]
        item_id = self.data.at[idx, "item_id"]
        rating = self.data.at[idx, "rating"]

        user_embed = self.user_embeds[user_id]
        item_embed = self.item_embeds[item_id]
        user_bias = self.user_bias[user_id]
        item_bias = self.item_bias[item_id]
        global_bias = self.global_bias

        return user_embed, item_embed, user_bias, item_bias, global_bias, rating