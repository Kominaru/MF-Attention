import argparse
import os
from dataset import DyadicRegressionDataModule

# Pre-constructs train-test splits for a dataset with the given number of splits and test set ratio

parser = argparse.ArgumentParser(description='Data Splitter')
parser.add_argument('--dataset', type=str, help='Path to the dataset')
parser.add_argument('--splits', type=int, help='Number of splits')
parser.add_argument('--test_ratio', type=float, help='Test set ratio')
args = parser.parse_args()

dataset_name = args.dataset
num_splits = args.splits
test_ratio = args.test_ratio

if __name__ == '__main__':

    # Create a directory to store the train-test splits
    splits_dir = f"data/{dataset_name}/splits/"
    os.makedirs(splits_dir, exist_ok=True)

    for i in range(1,num_splits+1):
        datamodule = DyadicRegressionDataModule(
            dataset_name,
            batch_size=2**15,
            num_workers=4,
            test_size=0.1,
            dataset_name=dataset_name,
            verbose=0
        )

        datamodule.train_dataset.data.to_csv(f"{splits_dir}/train_{i}.csv", index=False)
        datamodule.test_dataset.data.to_csv(f"{splits_dir}/test_{i}.csv", index=False)

        print(f"Split {i}: {len(datamodule.train_dataset)} train samples, {len(datamodule.test_dataset)} test samples")

    print('All splits created successfully.')