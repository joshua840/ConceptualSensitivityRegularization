import torch
from .models.load_model import load_model_head
from . import ERM


class CGR(ERM):
    def __init__(
        self,
        cgr_stage: str = "stage1",
        lamb_cs: float = 1,
        lamb_cav: float = 1,
        cs_method: str = "dot_sq",
        grad_from: str = "logit",
        target_layer: str = "classifier.1",
        g_num_classes: int = 1,
        g_model: str = "linear",
        g_criterion: str = "bce",
        g_activation: str = "softplus",
        g_softplus_beta: float = 10,
        g_freeze: bool = False,
        g_ckpt_path: str = None,
        **kwargs,
    ):
        """
        Args:
            cgr_stage: str, stage of cgr training
            lamb_cs: float, regularization constant for conceptual sensitivity loss
            lamb_cav: float, regularization constant for cav loss
            cs_method: str, Conceptual Sencitivity calculation method
            grad_from: str, gradient of logit vs loss
            target_layer: str, embedding layer
            g_num_classes: int, num_classes of model_g
            g_model: str, additional model
            g_criterion: str, criterion of model_g
            g_activation: str, activation of model_g
            g_softplus_beta: float, beta of model_g
            g_freeze: bool, criterion of model_g
            g_ckpt_path: str, ckpt path for model_g if needed
        """
        super().__init__(**kwargs)
        kwargs["module_name"] = "CGR" if cgr_stage == "stage2" else "CGRstage1"
        self.save_hyperparameters()

        self.in_features = {
            "resnet18": {
                "avgpool": 512,
                "layer4": 25088,
                "layer3": 50176,
            },
            "resnet50": {
                "avgpool": 2048,
            },
            "convnext_t": {
                "avgpool": 768,
                "classifier.1": 768,
            },
        }[self.hparams.model][self.hparams.target_layer]

        self.load_model_head()

        self.g_criterion_fn = self.get_criterion_fn(self.hparams.g_criterion)

    def load_model_head(self):
        self.model_g = load_model_head(
            model=self.hparams.g_model,
            activation_fn=self.hparams.g_activation,
            softplus_beta=self.hparams.g_softplus_beta,
            num_classes=self.hparams.g_num_classes,
            ckpt_path=self.hparams.g_ckpt_path,
            in_features=self.in_features,
            freeze=self.hparams.g_freeze,
        )

    def save_outputs_hook(self):
        def fn(_, __, output):
            self._hooked_features = output

        return fn

    def training_step(self, batch, batch_idx):
        if self.hparams.cgr_stage == "stage1":
            output = self.concept_step(batch, mode="train")
        elif self.hparams.cgr_stage == "stage2":
            output = self.classifier_step(batch, mode="train")

        return output

    def classifier_step(self, data, mode):
        x, y, g, i = data

        x.requires_grad = True if mode == "train" else False

        y_hat = self(x)
        pos_weight = torch.tensor([self.hparams.pos_weight], device=x.device)
        ce_loss = self.criterion_fn(y_hat, y, reduction="mean", pos_weight=pos_weight)
        loss = ce_loss

        if self.hparams.lamb_cs > 0 and mode == "train":
            # conceptual sensitivity
            if self.hparams.criterion == "bce":
                logit_c = y_hat * (-2 * y.float().unsqueeze(1) + 1)
            else:
                logit_c = y_hat[range(len(y)), y]

            if self.hparams.grad_from == "logit":
                grad_h = torch.autograd.grad(
                    outputs=logit_c.sum(),
                    inputs=x,
                    create_graph=True,
                    retain_graph=True,
                )[0]

                grad_g = torch.autograd.grad(
                    outputs=self.model_g(x).sum(),
                    inputs=x,
                    create_graph=True,
                    retain_graph=True,
                )[0]
            elif self.hparams.grad_from == "loss":
                grad_h = (
                    torch.autograd.grad(
                        outputs=ce_loss.sum(),
                        inputs=x,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    .squeeze()
                    .flatten(1)
                )
                logit_g = self.model_g(x)
                cav_ce_loss = self.g_criterion_fn(logit_g, y, reduction="mean")
                grad_g = (
                    torch.autograd.grad(
                        outputs=cav_ce_loss.sum(),
                        inputs=x,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    .squeeze()
                    .flatten(1)
                )

            if "cross" in self.hparams.cs_method:
                concept_gradient = grad_h @ grad_g.T
                feature_distance = (x @ x.T + 1) / 2  # [0, 1]

                if self.hparams.cs_method == "cross_dot_sq":
                    cs_loss = (concept_gradient.square() * feature_distance).mean()

            else:
                concept_gradient = torch.einsum("nc,nc->n", grad_h, grad_g)

                if self.hparams.cs_method == "dot_sq":
                    cs_loss = concept_gradient.square().mean()
                elif self.hparams.cs_method == "dot_abs":
                    cs_loss = concept_gradient.abs().mean()
                else:
                    concept_gradient = concept_gradient / (
                        (
                            grad_g.norm(dim=1, keepdim=True)
                            * grad_h.norm(dim=1, keepdim=True)
                        )
                        + 1e-7
                    )
                    if self.hparams.cs_method == "cosine_sq":
                        cs_loss = concept_gradient.square().mean()
                    elif self.hparams.cs_method == "cosine_abs":
                        cs_loss = concept_gradient.abs().mean()
                    elif self.hparams.cs_method == "cosine_log":
                        cs_loss = torch.log((1 - concept_gradient.abs()) + 1e-7).mean()

            loss = ce_loss + self.hparams.lamb_cs * cs_loss

        acc = self.metric.get_accuracy(y_hat, y, criterion=self.hparams.criterion)
        self.log_dict(
            {f"{mode}_loss": loss, f"{mode}_ce_loss": ce_loss, f"{mode}_acc": acc},
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        return {"loss": loss, "y_hat": y_hat.squeeze().detach(), "y": y, "g": g}

    def concept_step(self, data, mode):
        x, y, g, i = data

        x.requires_grad = False

        concept_logit = self.model_g(x)
        cav_ce_loss = self.g_criterion_fn(concept_logit, y, reduction="mean")

        loss = self.hparams.lamb_cav * cav_ce_loss

        acc = self.metric.get_accuracy(
            concept_logit, y, criterion=self.hparams.g_criterion
        )
        self.log_dict(
            {f"{mode}_cav_loss": cav_ce_loss, f"{mode}_cav_acc": acc},
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return loss

    def configure_optimizers(self):
        assert self.hparams.optimizer in [
            "sgd",
            "adam",
            "adamw",
        ], "Wrong optimizer name"

        if self.hparams.cgr_stage == "stage1":
            model = self.model_g
        elif self.hparams.cgr_stage == "stage2":
            model = self.model_h
        else:
            raise ValueError("Wrong cgr_stage")

        print(self.model_h.parameters())

        kwargs = {
            "lr": self.hparams.learning_rate,
            "weight_decay": self.hparams.weight_decay,
        }

        if self.hparams.optimizer == "sgd":
            optim = torch.optim.SGD(params=model.parameters(), momentum=0.9, **kwargs)
        elif self.hparams.optimizer == "adam":
            optim = torch.optim.Adam(params=model.parameters(), **kwargs)
        elif self.hparams.optimizer == "adamw":
            optim = torch.optim.AdamW(params=model.parameters(), **kwargs)

        MultiStepLR = torch.optim.lr_scheduler.MultiStepLR

        scheduler = {
            "scheduler": MultiStepLR(
                optimizer=optim, milestones=self.hparams.milestones, gamma=0.1
            ),
            "name": "lr_history1",
        }

        return [optim], [scheduler]
