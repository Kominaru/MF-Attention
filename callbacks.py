import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class CFValidationCallback(pl.callbacks.Callback):
    def __init__(self, cf_model, validation_dataloader, side = "user"):
        super().__init__()
        self.cf_model = cf_model
        self.val_dataloader = validation_dataloader
        self.side = side
        self.state = {"val_cf_rmse": []}

    def on_validation_epoch_end(self, trainer, pl_module):
        
        if pl_module.current_epoch % 25 != 0:
            return
       
        validation_outputs = pl_module.val_outputs
        validation_outputs = torch.cat(validation_outputs, dim=0)

        if self.side == "user":
            self.cf_model.user_embedding.weight.data = validation_outputs
        else:
            self.cf_model.item_embedding.weight.data = validation_outputs

        trainer = pl.Trainer(accelerator="auto", enable_progress_bar=False, gpus=1)
        cf_model_validation_loss = trainer.validate(self.cf_model, dataloaders=self.val_dataloader, verbose=False)[0]["val_rmse"]

        self.state["val_cf_rmse"].append(cf_model_validation_loss)

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
        