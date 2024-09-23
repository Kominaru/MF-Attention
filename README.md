<h1 style="text-align: center;">MF-Attention</h1>

This project contains Work in Progress utilities for utilizing attention-based mechanisms leveraging Dynamic Data Selection to compress user and item embeddings in Recommender Systems applications. 

The objective of using this method is two-fold:
- Using DDS-based attention allows the compression mechanism to be aware of user characteristics before compression
- Create a reusable compressor: not only it can be used to compress and decompress the embeddings of the entities it was trained with, but also of new embeddings trained by the same overarching RS system (e.g. when new users or clients are added to the system).
- Is not affected by popularity/representation bias of the overarching MF data or model (as the compressor is trained on embedding reconstruction, rather than the MF task data itself).


> [!CAUTION]
> This project is a very early Work in Progress and may not be fully functional, change or break at any time without warning.

## Architecture

<p align="center">
    <img src="https://imgur.com/tDCIjpy.png" alt="Architecture Image">
</p>

## Setup

The project is built using Python 3.10 and untested on other versions. It is recommended to use a virtual environment to install the dependencies using `pip install -r requirements.txt`.

The project is structured as follows:
- `compressor/`: Contains the implementation of the DDS-based attention compressor
- `mf/`: Contains the implementation of a basic Matrix Factorization model for Recommender System tasks, for testing the compressor.
- `data/`: Contains datasets and data processing utilities for using with the compressor and the RS models. 

- `main_compressor.py`: Contains the main script for training and testing the compressor on specified trained RS model.
- `main_mf.py`: Contains the main script for training and testing an MF model on the specified dataset and dataset split.

Currently, this repository does not include the scripts for downloading, preprocessing and splitting the datasets.

## Data
So far, this project has been developed with Rating Prediction tasks in mind, though the compressor is agnostic to the specific RS task the embeddings are used for. The datasets used have been:
- [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- [MovieLens 10M](https://grouplens.org/datasets/movielens/10m/)
- [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)
- [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)
- [Netflix Prize](https://www.kaggle.com/netflix-inc/netflix-prize-data)
- Douban Monti

