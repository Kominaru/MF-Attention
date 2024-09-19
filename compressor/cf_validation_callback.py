import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

DATASET_RANGES = {"ml-1m": (0.835, 0.860), "ml-10m": (0.765, 0.790), "ml-25m": (0.740, 0.765)}


class CFValidationCallback(pl.callbacks.Callback):
    def __init__(self, cf_model, validation_datamodule, embeddings_datamodule=None):
        super().__init__()
        self.cf_model = cf_model
        self.val_datamodule = validation_datamodule
        self.state = {
            "val_cf_rmse_train": [],
            "val_cf_rmse_val": [],
            "val_cf_rmse": [],
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
        }
        self.dataset = validation_datamodule.dataset
        self.split = validation_datamodule.split
        self.embeddings_datamodule = embeddings_datamodule

    def on_validation_epoch_end(self, trainer, pl_module):

        self.state["train_loss"].append(trainer.callback_metrics["val_loss/dataloader_idx_0"].item())
        self.state["val_loss"].append(trainer.callback_metrics["val_loss/dataloader_idx_1"].item())

        if not (
            (pl_module.current_epoch < 500 and pl_module.current_epoch % 50 == 0)
            or (pl_module.current_epoch >= 500 and pl_module.current_epoch % 250 == 0)
        ):
            return

        validation_outputs = torch.cat(pl_module.val_outputs, dim=0).cpu()

        if self.embeddings_datamodule.entity_type == "user":
            self.cf_model.user_embedding.weight.data[self.embeddings_datamodule.id_order] = validation_outputs
        else:
            self.cf_model.item_embedding.weight.data[self.embeddings_datamodule.id_order] = validation_outputs

        trainer_cf = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)
        cf_model_validation_losses = trainer_cf.validate(
            self.cf_model,
            dataloaders=self.val_datamodule.val_dataloader(self.embeddings_datamodule.entity_type),
            verbose=False,
        )

        pl_module.log(
            "val_cf_rmse_train",
            cf_model_validation_losses[0]["val_rmse/dataloader_idx_0"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            "val_cf_rmse_val",
            cf_model_validation_losses[1]["val_rmse/dataloader_idx_1"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        pl_module.log(
            "val_cf_rmse", cf_model_validation_losses[0]["val_rmse"], on_step=False, on_epoch=True, prog_bar=True
        )

        self.state["val_cf_rmse_train"].append(cf_model_validation_losses[0]["val_rmse/dataloader_idx_0"])
        self.state["val_cf_rmse_val"].append(cf_model_validation_losses[1]["val_rmse/dataloader_idx_1"])
        self.state["val_cf_rmse"].append(cf_model_validation_losses[0]["val_rmse"])

        self.state["epoch"].append(pl_module.current_epoch)

        pl_module.val_outputs = []

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state = state_dict

    def on_train_end(self, trainer, pl_module):

        plt.plot(
            self.state["epoch"],
            self.state["val_cf_rmse_train"],
            label=f"Compressor train {self.embeddings_datamodule.entity_type}s",
            color="red",
        )
        plt.plot(
            self.state["epoch"],
            self.state["val_cf_rmse_val"],
            label=f"Compressor val {self.embeddings_datamodule.entity_type}s",
            color="blue",
        )

        plt.xlabel("Compressor Epoch")
        plt.ylabel("MF Val score (RMSE)")

        plt.legend(loc="upper left")

        plt.ylim(DATASET_RANGES[self.dataset])

        plt.twinx()

        plt.plot(
            [i for i in range(50, len(self.state["train_loss"]))],
            self.state["train_loss"][50:],
            label="Compressor train {self.embeddings_datamodule.entity_type}s",
            color="red",
            alpha=0.5,
        )
        plt.plot(
            [i for i in range(50, len(self.state["val_loss"]))],
            self.state["val_loss"][50:],
            label="Compressor val {self.embeddings_datamodule.entity_type}s",
            color="blue",
            alpha=0.5,
        )

        plt.ylabel("Compressor reconstruction loss (MSE)")

        plt.title(
            f"MF Val set scores and Compressor reconstruction loss ({self.embeddings_datamodule.entity_type} compression)"
        )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(
            f"figures/compressor_training/{self.dataset}/{self.embeddings_datamodule.entity_type}s_split{self.split}.pdf"
        )
        plt.clf()
