import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from dataset import DyadicRegressionDataset

DATASET_RANGES = {
    "ml-1m": (0.835, 0.860),
    "ml-10m": (0.765, 0.790),
    "ml-25m": (0.740, 0.765)
}


class CFValidationCallback(pl.callbacks.Callback):
    def __init__(self, cf_model, validation_dataloaders, side="user", ids=None, dataset=None):
        super().__init__()
        self.cf_model = cf_model
        self.val_dataloader = validation_dataloaders
        self.side = side
        self.state = {"val_cf_rmse_known": [], "val_cf_rmse_unknown": [], "val_cf_rmse": [], "epoch": [], "known_loss": [], "unknown_loss": []}
        self.ids = ids
        self.dataset = dataset

    def on_validation_epoch_end(self, trainer, pl_module):

        self.state["known_loss"].append(trainer.callback_metrics["val_loss/dataloader_idx_0"].item())
        self.state["unknown_loss"].append(trainer.callback_metrics["val_loss/dataloader_idx_1"].item())

        if not (
            (pl_module.current_epoch < 500 and pl_module.current_epoch % 50 == 0)
            or (pl_module.current_epoch >= 500 and pl_module.current_epoch % 250 == 0)
        ):
            return

        validation_outputs = pl_module.val_outputs
        validation_outputs = torch.cat(validation_outputs, dim=0)

        # Expand dims on the first axis to match the embedding shape
        # print(self.ids.shape, validation_outputs.shape)

        if self.side == "user":
            self.cf_model.user_embedding.weight.data[self.ids] = validation_outputs.cpu()
        else:
            self.cf_model.item_embedding.weight.data[self.ids] = validation_outputs.cpu()

        trainer_cf = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)
        cf_model_validation_losses = trainer_cf.validate(self.cf_model, dataloaders=self.val_dataloader, verbose=False)

        if len(cf_model_validation_losses) == 2:

            pl_module.log(
                "val_cf_rmse_known",
                cf_model_validation_losses[0]["val_rmse/dataloader_idx_0"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            pl_module.log(
                "val_cf_rmse_unknown",
                cf_model_validation_losses[1]["val_rmse/dataloader_idx_1"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        pl_module.log(
            "val_cf_rmse", cf_model_validation_losses[0]["val_rmse"], on_step=False, on_epoch=True, prog_bar=True
        )

        if len(cf_model_validation_losses) == 2:

            self.state["val_cf_rmse_known"].append(cf_model_validation_losses[0]["val_rmse/dataloader_idx_0"])
            self.state["val_cf_rmse_unknown"].append(cf_model_validation_losses[1]["val_rmse/dataloader_idx_1"])
        self.state["val_cf_rmse"].append(cf_model_validation_losses[0]["val_rmse"])

        self.state["epoch"].append(pl_module.current_epoch)

        pl_module.val_outputs = []

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state = state_dict

    def on_train_end(self, trainer, pl_module):

        # plt.plot(self.state["epoch"], self.state["val_cf_rmse"])
        if len(self.state["val_cf_rmse_known"]) > 0:
            plt.plot(self.state["epoch"][15:], self.state["val_cf_rmse_known"][15:], label=f"Known {self.side}s", color="red")
            plt.plot(self.state["epoch"][15:], self.state["val_cf_rmse_unknown"][15:], label=f"Unknown {self.side}s", color="blue")
            # plt.plot(self.state["epoch"][15:], self.state["val_cf_rmse"][15:], label=f"All {self.side}s", color="purple")

        plt.xlabel("Compressor Epoch")
        plt.ylabel("Validation RMSE")

        plt.legend(loc="upper left")

        plt.ylim(DATASET_RANGES[self.dataset])


        plt.twinx()

        plt.plot([i for i in range(50,len(self.state["known_loss"]))], self.state["known_loss"][50:], label="Known Loss", color="red", alpha=0.5)
        plt.plot([i for i in range(50,len(self.state["unknown_loss"]))], self.state["unknown_loss"][50:], label="Unknown Loss", color="blue", alpha=0.5)

        plt.ylabel("Reconstruction Loss")

        plt.title(f"Validation RMSE of the Compressed Model ({self.side})")
        plt.legend(loc = "upper right")
        plt.tight_layout()
        plt.savefig(f"compressor_data/{self.dataset}/{self.side}_validation_rmse.pdf")
        plt.clf()