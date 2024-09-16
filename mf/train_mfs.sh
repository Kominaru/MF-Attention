# Train all 5 splits for the given datasets and embedding dims

# Usage: bash train_mfs.sh

datasets=("ml-25m")  # List of datasets
embedding_dims=(32 512)  # List of embedding dimensions
splits=5  # Number of splits

for dataset in "${datasets[@]}"; do
    for embedding_dim in "${embedding_dims[@]}"; do
        for ((split=1; split<=$splits; split++)); do
            echo "Training model for dataset: $dataset, embedding_dim: $embedding_dim, split: $split"
            python main.py --dataset $dataset --d $embedding_dim --split $split 
        done
    done
done