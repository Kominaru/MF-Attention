import os
from datetime import datetime

def save_model_outputs(train_df, test_df, model_name, dataset_name, model_params):
    # Create outputs/{model_name} folder if it doesn't exist
    if not os.path.exists(f"outputs/{dataset_name}/{model_name}"):
        os.makedirs(f"outputs/{dataset_name}/{model_name}")
    
    # Get list of files in outputs/{model_name} folder
    files = os.listdir(f"outputs/{dataset_name}/{model_name}")
    
    # Filter list of files to only include train_outputs*.csv and test_outputs*.csv files
    files = [f for f in files if f.startswith("train_outputs") or f.startswith("test_outputs")]
    
    # Get the maximum integer value in the filenames
    if len(files) == 0:
        i = 1
    else:
        i = max([int(f.split(".")[0].split("_")[-1]) for f in files]) + 1
    
    # Save train_df and test_df to files
    train_df.to_csv(f"outputs/{dataset_name}/{model_name}/train_outputs_{i}.csv", index=False)
    test_df.to_csv(f"outputs/{dataset_name}/{model_name}/test_outputs_{i}.csv", index=False)
    
    # Write model_params to params.txt file
    with open(f"outputs/{dataset_name}/{model_name}/params.txt", "a") as f:
        s = "\t".join([str(i), str(datetime.now())]+[f"{k}: {v}" for k, v in model_params.items()])
        f.write(s+"\n")
