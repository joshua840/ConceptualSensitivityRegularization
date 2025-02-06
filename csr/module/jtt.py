import torch
import os
from .erm import ERM
from lightning.pytorch.callbacks import ModelCheckpoint


class JTTMetadataGenerator(ERM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs["module_name"] = "JTTMeta"
        self.save_hyperparameters()

    def validation_step(self, batch, batch_idx):
        outputs = self.default_step(batch, mode=f"valid")
        self.log("valid_valid_worst_acc", 0)
        self.validation0_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        if self.current_epoch == 0:
            self.validation0_step_outputs = []
            return
        outputs = self.validation0_step_outputs

        y_hat = torch.cat([elem["y_hat"] for elem in outputs])
        y = torch.cat([elem["y"] for elem in outputs])

        incorrects = (y_hat > 0) != (y > 0)
        incorrect_indices = torch.where(incorrects)[0].detach()

        callbacks = [
            elem for elem in self.trainer.callbacks if isinstance(elem, ModelCheckpoint)
        ][0]

        save_path = os.path.join(callbacks.dirpath, f"jtt_meta_{self.current_epoch}.pt")
        # if directory is not exist, create it
        if not os.path.exists(callbacks.dirpath):
            os.makedirs(callbacks.dirpath)

        torch.save(incorrect_indices, save_path)

        self.validation0_step_outputs = []
        return

    def test_epoch_end(self, outputs):
        return


class JTT(ERM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs["module_name"] = "JTT"
        self.save_hyperparameters()
