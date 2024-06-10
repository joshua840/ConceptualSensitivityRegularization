import torch
from . import CGR


class ProjCGR(CGR):
    def __init__(
        self,
        proj_cav_ckpt_path: str = None,
        **kwargs,
    ):
        """
        Args:
            proj_cav_ckpt_path: str, ckpt path for model_cav if needed
        """
        super().__init__(**kwargs)
        kwargs["module_name"] = "PCGR"
        self.save_hyperparameters()
        self.cav = self.load_model_cav()

    def load_model_cav(self):
        cav = torch.load(self.hparams.proj_cav_ckpt_path)
        cav = cav["state_dict"]["model_g.1.weight"].cuda()

        return cav / torch.norm(cav)

    def default_step(self, batch, mode):
        x, y, g, i = batch
        batch = self.projection(x), y, g, i

        return super().default_step(batch, mode)

    def projection(self, x):
        return x - torch.matmul(x, self.cav.T) * self.cav

    def training_step(self, batch, batch_idx):
        x, y, g, i = batch
        batch = self.projection(x), y, g, i
        return super().classifier_step(batch, mode="train")
