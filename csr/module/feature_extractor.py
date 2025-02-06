import torch
from .erm import ERM
import os


class Fhook:
    def __init__(self):
        pass

    def save_outputs_hook(self):
        def fn(_, __, output):
            self._hooked_features = output.detach().cpu()

        return fn


class FeatureExtractor(ERM):
    def __init__(self, save_root: str, target_layer: str, **kwargs):
        """
        Args:
            save_root: root of the directory to save features
            target_layer: layer to be hooked
        """
        super().__init__(**kwargs)

        self.hook = Fhook()
        for name, module in self.named_modules():
            if name == "model." + self.hparams.target_layer:
                self.handle = module.register_forward_hook(
                    self.hook.save_outputs_hook()
                )
                print("hook is succesfully registered")

        self.outputs = {}

        for mode in ["tr", "va", "te"]:
            self.outputs[mode] = {"features": [], "ys": [], "gs": []}

        kwargs["module_name"] = "FeatureGenerator"
        self.save_hyperparameters()

    def on_train_epoch_start(self):
        # reinit outputs
        self.outputs["tr"] = {"features": [], "ys": [], "gs": []}

    def on_train_epoch_end(self):
        self.save_features(mode="tr")

    def save_features(self, mode):
        results_dict = self.outputs[mode]
        features = torch.cat(results_dict["features"], dim=0).half()

        dirs = os.path.join(
            self.hparams.save_root, self.hparams.dataset, self.hparams.model, mode
        )

        os.makedirs(dirs, exist_ok=True)
        torch.save(features, os.path.join(dirs, f"{self.current_epoch}.pt"))

        if self.current_epoch == 0:
            ys = torch.cat(results_dict["ys"], dim=0)
            gs = torch.cat(results_dict["gs"], dim=0)
            torch.save([ys, gs, range(len(gs))], os.path.join(dirs, f"metadata.pt"))

    def on_validation_epoch_end(self):
        if self.current_epoch != 0:
            return
        self.save_features(mode="va")
        self.save_features(mode="te")

    def default_step(self, batch, mode):
        x, y, g, i = batch
        self(x)

        results_dict = self.outputs[mode]

        results_dict["features"].append(
            torch.flatten(self.hook._hooked_features, start_dim=1).detach().cpu()
        )

        if self.current_epoch == 0:
            results_dict["ys"].append(y.detach().cpu())
            results_dict["gs"].append(g.detach().cpu())

        return

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if self.current_epoch != 0:
            return
        if dataloader_idx == 0:
            self.default_step(batch, mode=f"va")
        elif dataloader_idx == 1:
            self.default_step(batch, mode=f"te")

    def training_step(self, batch, batch_idx):
        self.default_step(batch, mode=f"tr")
