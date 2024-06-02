import logging
import os
import numpy as np
from models.collaborativefiltering_mf import CollaborativeFilteringModel
import pandas as pd
from dataset import FastTensorDataLoader, DyadicRegressionDataset, DyadicRegressionDataModule
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(logging.WARNING)

TEACHER_DIM = 512
STUDENT_DIM = 8

if __name__ == "__main__":

    datamodule = DyadicRegressionDataModule(
        "data",
        batch_size=2**15,
        num_workers=4,
        test_size=0.1,
        dataset_name="ml-10m",
        verbose=0
    )

    model_teacher = CollaborativeFilteringModel.load_from_checkpoint(
        f"models/MF/checkpoints/best-model-{TEACHER_DIM}.ckpt"
    )

    model_student = CollaborativeFilteringModel(
        num_users=datamodule.num_users,
        num_items=datamodule.num_items,
        embedding_dim=STUDENT_DIM,
        lr=5e-4,
        l2_reg=1e-2,
        rating_range=(datamodule.min_rating, datamodule.max_rating),
        use_biases=False,
        activation="sigmoid",
        sigmoid_scale=1.0
    )

    teacher_train_data = pd.read_csv(f"compressor_data/_train_{TEACHER_DIM}.csv")
    teacher_test_data = pd.read_csv(f"compressor_data/_test_{TEACHER_DIM}.csv")

    trainer = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)

    teacher_train_preds = np.concatenate(
        trainer.predict(model_teacher, dataloaders=DataLoader(
        DyadicRegressionDataset(teacher_train_data), batch_size=2**14, shuffle=False, num_workers=4, persistent_workers=True
    ))
    )

    teacher_train_data["rating"] = teacher_train_preds

    datamodule.train_df = teacher_train_data
    datamodule.test_df = teacher_test_data
    
    test_dataloader = DataLoader(
        DyadicRegressionDataset(teacher_test_data), batch_size=2**14, shuffle=False, num_workers=4, persistent_workers=True
    )

    callbacks: list = [
    pl.callbacks.ModelCheckpoint(
            dirpath="models/MF/checkpoints",
            filename=f"student-model-{STUDENT_DIM}",
            monitor="val_rmse",
            mode="min",
        ), 

    pl.callbacks.EarlyStopping(
        monitor="val_rmse",
        mode="min",
        patience=5,
        verbose=False,
        min_delta=0.0001,
    )
    ]

    if os.path.exists(f"models/MF/checkpoints/student-model-{STUDENT_DIM}.ckpt"):
        os.remove(f"models/MF/checkpoints/student-model-{STUDENT_DIM}.ckpt")
    
    trainer = pl.Trainer(
        gpus=1,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=False,
        precision=16,
        enable_model_summary=True,
        enable_progress_bar=True,
    )

    trainer.fit(model_student, datamodule=datamodule)

    student_model = CollaborativeFilteringModel.load_from_checkpoint(f"models/MF/checkpoints/student-model-{STUDENT_DIM}.ckpt")

    student_test_preds = np.concatenate(
        trainer.predict(student_model, dataloaders=test_dataloader)
    )

    student_test_rmse = np.sqrt(np.mean((student_test_preds - teacher_test_data["rating"].values) ** 2))

    print(f"\tStudent Test RMSE: {student_test_rmse}")











    


    

    
