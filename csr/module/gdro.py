import torch
from .erm import ERM


class GDRO(ERM):
    def __init__(self, eta: float = 1, **kwargs):
        """
        Args:
            eta: float, hyperparameter for group DRO
        """
        super().__init__(**kwargs)
        self.register_buffer("q", torch.ones(self.hparams.num_groups))
        kwargs["module_name"] = "GDRO"
        self.save_hyperparameters()

    def groups_(self, y, g):
        idx_g, idx_b = [], []
        all_g = g * self.hparams.num_classes + y

        for g in all_g.unique():
            idx_g.append(g.long())
            idx_b.append(all_g == g)

        return zip(idx_g, idx_b)

    def default_step(self, batch, mode):
        x, y, g, i = batch

        y_hat = self(x)
        pos_weight = torch.tensor([self.hparams.pos_weight], device=x.device)
        losses = self.criterion_fn(y_hat, y, reduction="none", pos_weight=pos_weight)

        for idx_g, idx_b in self.groups_(y, g):
            self.q[idx_g] *= (self.hparams.eta * losses[idx_b].mean()).exp().item()

        self.q /= self.q.sum() + 1e-8

        loss = 0
        for idx_g, idx_b in self.groups_(y, g):
            loss += self.q[idx_g] * losses[idx_b].mean()

        acc = self.metric.get_accuracy(y_hat, y, criterion=self.hparams.criterion)
        self.log_dict(
            {f"{mode}_loss": loss, f"{mode}_acc": acc},
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        return {"loss": loss, "y_hat": y_hat.squeeze().detach(), "y": y, "g": g}
