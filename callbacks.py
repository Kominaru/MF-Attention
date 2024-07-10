import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from dataset import DyadicRegressionDataset


class CFValidationCallback(pl.callbacks.Callback):
    def __init__(self, cf_model, validation_dataloaders, side="user", ids=None, dataset=None):
        super().__init__()
        self.cf_model = cf_model
        self.val_dataloader = validation_dataloaders
        self.side = side
        self.state = {"val_cf_rmse_known": [], "val_cf_rmse_unknown": [], "val_cf_rmse": [], "epoch": []}
        self.ids = ids
        self.dataset = dataset

    def on_validation_epoch_end(self, trainer, pl_module):

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

        plt.plot(self.state["epoch"], self.state["val_cf_rmse"])
        if len(self.state["val_cf_rmse_known"]) > 0:
            plt.plot(self.state["epoch"], self.state["val_cf_rmse_known"], label=f"Known {self.side}s", color="red")
            plt.plot(
                self.state["epoch"], self.state["val_cf_rmse_unknown"], label=f"Unknown {self.side}s", color="blue"
            )
            plt.plot(self.state["epoch"], self.state["val_cf_rmse"], label=f"All {self.side}s", color="purple")
        plt.xlabel("Compressor Epoch")
        plt.ylabel("Validation RMSE")

        plt.ylim(0.742, 0.770)
        # Tick every 0.001 increment
        plt.yticks([i / 1000 for i in range(742, 770, 2)])
        plt.grid(axis="y", alpha=0.2, zorder=0)

        plt.title(f"Validation RMSE of the Compressed Model ({self.side})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"compressor_data/{self.dataset}/{self.side}_validation_rmse.pdf")
        plt.clf()


class CFValidationCallbackSimultaneous(pl.callbacks.Callback):
    def __init__(self, cf_model, validation_data, ids=None):
        super().__init__()
        self.cf_model = cf_model
        self.val_data = validation_data
        self.state = {
            "val_cf_rmse_user": [],
            "val_cf_rmse_item": [],
            "val_cf_rmse": [],
            "val_cf_rmse_og": [],
            "epoch": [],
            "val_cf_rmse_ui": [],
            "val_cf_rmse_u": [],
            "val_cf_rmse_i": [],
            "val_cf_rmse_new": [],
        }
        self.user_ids, self.item_ids = ids
        self.val_percent = 0.1

        self.known_users = self.user_ids[: (len(self.user_ids) - int(len(self.user_ids) * self.val_percent))]
        self.unknown_users = self.user_ids[(len(self.user_ids) - int(len(self.user_ids) * self.val_percent)) :]

        self.known_items = self.item_ids[: (len(self.item_ids) - int(len(self.item_ids) * self.val_percent))]
        self.unknown_items = self.item_ids[(len(self.item_ids) - int(len(self.item_ids) * self.val_percent)) :]

        self.val_dataloaders = [
            torch.utils.data.DataLoader(
                DyadicRegressionDataset(
                    self.val_data[
                        self.val_data["user_id"].isin(self.known_users)
                        & self.val_data["item_id"].isin(self.known_items)
                    ].reset_index(drop=True)
                ),
                batch_size=2**14,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
            ),
            torch.utils.data.DataLoader(
                DyadicRegressionDataset(
                    self.val_data[
                        self.val_data["user_id"].isin(self.known_users)
                        & self.val_data["item_id"].isin(self.unknown_items)
                    ].reset_index(drop=True)
                ),
                batch_size=2**14,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
            ),
            torch.utils.data.DataLoader(
                DyadicRegressionDataset(
                    self.val_data[
                        self.val_data["user_id"].isin(self.unknown_users)
                        & self.val_data["item_id"].isin(self.known_items)
                    ].reset_index(drop=True)
                ),
                batch_size=2**14,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
            ),
            torch.utils.data.DataLoader(
                DyadicRegressionDataset(
                    self.val_data[
                        self.val_data["user_id"].isin(self.unknown_users)
                        & self.val_data["item_id"].isin(self.unknown_items)
                    ].reset_index(drop=True)
                ),
                batch_size=2**14,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
            ),
        ]

        print("Validation dataloaders created")
        print("Both known:", len(self.val_dataloaders[0].dataset))
        print("User known:", len(self.val_dataloaders[1].dataset))
        print("Item known:", len(self.val_dataloaders[2].dataset))
        print("Both unknown:", len(self.val_dataloaders[3].dataset))

    def on_validation_epoch_end(self, trainer, pl_module):

        if not (
            (pl_module.current_epoch < 500 and pl_module.current_epoch % 50 == 0)
            or (pl_module.current_epoch >= 500 and pl_module.current_epoch % 250 == 0)
        ):
            return

        validation_outputs = pl_module.val_outputs
        validation_outputs = torch.cat(validation_outputs, dim=0)

        user_outputs = validation_outputs[: len(self.user_ids)]
        item_outputs = validation_outputs[len(self.user_ids) :]

        original_user_embedding = self.cf_model.user_embedding.weight.data.clone()
        original_item_embedding = self.cf_model.item_embedding.weight.data.clone()

        self.cf_model.user_embedding.weight.data[self.user_ids] = user_outputs.cpu()
        self.cf_model.item_embedding.weight.data[self.item_ids] = item_outputs.cpu()

        # trainer_cf = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)
        # cf_user_rmse = trainer_cf.validate(self.cf_model, dataloaders=self.val_dataloader, verbose=False)[0]["val_rmse"]
        # pl_module.log("val_cf_rmse_user", cf_user_rmse, on_step=False, on_epoch=True, prog_bar=True)

        # self.cf_model.user_embedding.weight.data = original_user_embedding
        # self.cf_model.item_embedding.weight.data[self.item_ids] = item_outputs.cpu()

        # trainer_cf = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)
        # cf_item_rmse = trainer_cf.validate(self.cf_model, dataloaders=self.val_dataloader, verbose=False)[0]["val_rmse"]
        # pl_module.log("val_cf_rmse_item", cf_item_rmse, on_step=False, on_epoch=True, prog_bar=True)

        # self.cf_model.user_embedding.weight.data[self.user_ids] = user_outputs.cpu()
        # trainer_cf = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)
        # cf_both_rmse = trainer_cf.validate(self.cf_model, dataloaders=self.val_dataloader, verbose=False)[0]["val_rmse"]
        # pl_module.log("val_cf_rmse", cf_both_rmse, on_step=False, on_epoch=True, prog_bar=True)

        # self.state["val_cf_rmse"].append(cf_both_rmse)
        # self.state["val_cf_rmse_user"].append(cf_user_rmse)
        # self.state["val_cf_rmse_item"].append(cf_item_rmse)

        trainer_cf = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)
        cf_model_validation_losses = trainer_cf.validate(
            self.cf_model, dataloaders=self.val_dataloaders, verbose=False
        )

        pl_module.log(
            "val_cf_rmse_ui",
            cf_model_validation_losses[0]["val_rmse/dataloader_idx_0"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            "val_cf_rmse_u",
            cf_model_validation_losses[1]["val_rmse/dataloader_idx_1"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            "val_cf_rmse_i",
            cf_model_validation_losses[2]["val_rmse/dataloader_idx_2"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            "val_cf_rmse_new",
            cf_model_validation_losses[3]["val_rmse/dataloader_idx_3"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            "val_cf_rmse", cf_model_validation_losses[0]["val_rmse"], on_step=False, on_epoch=True, prog_bar=True
        )

        self.state["val_cf_rmse_ui"].append(cf_model_validation_losses[0]["val_rmse/dataloader_idx_0"])
        self.state["val_cf_rmse_u"].append(cf_model_validation_losses[1]["val_rmse/dataloader_idx_1"])
        self.state["val_cf_rmse_i"].append(cf_model_validation_losses[2]["val_rmse/dataloader_idx_2"])
        self.state["val_cf_rmse_new"].append(cf_model_validation_losses[3]["val_rmse/dataloader_idx_3"])
        self.state["val_cf_rmse"].append(cf_model_validation_losses[0]["val_rmse"])

        self.state["epoch"].append(pl_module.current_epoch)

        self.cf_model.user_embedding.weight.data = original_user_embedding
        self.cf_model.item_embedding.weight.data = original_item_embedding

        pl_module.val_outputs = []

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state = state_dict

    def on_train_end(self, trainer, pl_module):

        # plt.plot(self.state["epoch"], self.state["val_cf_rmse"], label="Both")
        # plt.plot(self.state["epoch"], self.state["val_cf_rmse_user"] , label="User")
        # plt.plot(self.state["epoch"], self.state["val_cf_rmse_item"] , label="Item")
        plt.plot(self.state["epoch"], self.state["val_cf_rmse_ui"], label="Both Known", color="purple")
        plt.plot(self.state["epoch"], self.state["val_cf_rmse_u"], label="User Known", color="blue")
        plt.plot(self.state["epoch"], self.state["val_cf_rmse_i"], label="Item Known", color="red")
        plt.plot(self.state["epoch"], self.state["val_cf_rmse_new"], label="Both Unknown", color="black")
        plt.plot(self.state["epoch"], self.state["val_cf_rmse"], label="All", color="green")

        plt.xlabel("Compressor Epoch")
        plt.ylabel("Validation RMSE")

        plt.ylim(0.765, 0.81)
        # Tick every 0.001 increment
        plt.yticks([i / 1000 for i in range(765, 810, 2)])
        plt.grid(axis="y", alpha=0.2, zorder=0)

        plt.title(f"Validation RMSE of the Compressed Model (Both)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"compressor_data/both_validation_rmse.pdf")
