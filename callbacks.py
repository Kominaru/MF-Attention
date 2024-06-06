import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class CFValidationCallback(pl.callbacks.Callback):
    def __init__(self, cf_model, validation_dataloaders, side = "user", ids = None):
        super().__init__()
        self.cf_model = cf_model
        self.val_dataloader = validation_dataloaders
        self.side = side
        self.state = {"val_cf_rmse_known": [], "val_cf_rmse_unknown": [], "val_cf_rmse": []}
        self.ids = ids

    def on_validation_epoch_end(self, trainer, pl_module):
        
        if pl_module.current_epoch % 25 != 0:
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

            pl_module.log("val_cf_rmse_known", cf_model_validation_losses[0]["val_rmse/dataloader_idx_0"], on_step=False, on_epoch=True, prog_bar=True)
            pl_module.log("val_cf_rmse_unknown", cf_model_validation_losses[1]["val_rmse/dataloader_idx_1"], on_step=False, on_epoch=True, prog_bar=True)

        pl_module.log("val_cf_rmse", cf_model_validation_losses[0]["val_rmse"], on_step=False, on_epoch=True, prog_bar=True)

        if len(cf_model_validation_losses) == 2:

                self.state["val_cf_rmse_known"].append(cf_model_validation_losses[0]["val_rmse/dataloader_idx_0"])
                self.state["val_cf_rmse_unknown"].append(cf_model_validation_losses[1]["val_rmse/dataloader_idx_1"])
        self.state["val_cf_rmse"].append(cf_model_validation_losses[0]["val_rmse"])

        pl_module.val_outputs = []

    def state_dict(self):
        return self.state
    
    def load_state_dict(self, state_dict):
        self.state = state_dict
    
    def on_train_end(self, trainer, pl_module):
        
        epochs = [i*25 for i in range(len(self.state["val_cf_rmse"]))]
        plt.plot(epochs, self.state["val_cf_rmse"])
        plt.xlabel("Compressor Epoch")
        plt.ylabel("Validation RMSE")

        plt.ylim(0.75, 0.85)

        plt.title(f"Validation RMSE of the Compressed Model ({self.side})")
        plt.tight_layout()
        plt.savefig(f"compressor_data/{self.side}_validation_rmse.pdf")
        