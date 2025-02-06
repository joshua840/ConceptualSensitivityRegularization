import torch
import torch.nn.functional as F

from .models import load_model
from .utils.metrics import Metrics
from .datamodule import DataModule

import typing


class ERM(DataModule):
    """
    `lightning.pytorch.Trainer` will call the functions by the following order

    for epoch in epochs:
        for batch in data:
            on_train_batch_start()
            for opt in optimizers:
                loss = train_step(batch, batch_idx, optimizer_idx)
                opt.zero_grad()
                loss.backward()
                opt.step()

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()
    """

    def __init__(
        self,
        model,
        input_type: str = "feature",
        imagenet_pretrained: bool = True,
        model_path: str = None,
        freeze_model: bool = True,
        freezing_target_layer: str = None,
        set_last_layer: str = "linear",
        del_backbone: bool = True,
        h_activation_fn: str = "softplus",
        h_softplus_beta: float = 10,
        learning_rate: float = 1e-3,
        scheduler: str = "multistep",
        milestones: list = [9999],
        max_epochs: int = 200,
        weight_decay: float = 1e-2,
        optimizer: str = "adamw",
        criterion: str = "bce",
        **kwargs,
    ):
        """
        Args:
            model: model to be used
            imagenet_pretrained: load default pretrained model
            model_path: A path of saved parameter
            freeze_model: freeze parameters or not
            freezing_target_layer: freeze parameters or not
            set_last_layer: architectures to replace last layer
            del_backbone: delete backbone or not
            input_type: raw or feature
            h_activation_fn: activation functions of model
            h_softplus_beta: beta of softplus
            learning_rate: learning rate of optimizer
            scheduler: lr schedular
            milestones: lr schedular (multistep)
            max_epochs: lr schedular (cosineannealing)
            weight_decay: weight decay of optimizer
            optimizer: optimizer to be used
            criterion: determine loss
            pos_weight: neg_samples/pos_samples
        """
        super().__init__(**kwargs)
        kwargs["module_name"] = "ERM"
        self.save_hyperparameters()

        self.model = load_model(
            model=self.hparams.model,
            num_classes=(
                1 if self.hparams.criterion == "bce" else self.hparams.num_classes
            ),
            imagenet_pretrained=self.hparams.imagenet_pretrained,
            model_h_activation_fn=self.hparams.h_activation_fn,
            model_h_softplus_beta=self.hparams.h_softplus_beta,
            model_path=self.hparams.model_path,
            freeze_model=self.hparams.freeze_model,
            last_layer=self.hparams.set_last_layer,
            freezing_target_layer=self.hparams.freezing_target_layer,
        )

        if self.hparams.input_type == "feature":
            self.model_h = (
                self.model.fc
                if self.hparams.model in ["resnet18", "resnet50"]
                else self.model.classifier[2]
            )

            if not self.hparams.freeze_model:
                assert not self.hparams.del_backbone
            if self.hparams.del_backbone:
                del self.model

        # For managing metrics
        self.metric = Metrics()
        self.criterion_fn = self.get_criterion_fn(self.hparams.criterion)

        # For convinient hyperparameter tuning
        self.best_worst_val_loss = 1e6
        self.best_epoch = 0.0
        self.best_worst_acc = 0.0
        self.best_wc_epoch = 0.0
        self.best_wc_acc = 0.0

        # For saving outputs
        self.validation0_step_outputs = []
        self.validation1_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        if self.hparams.input_type == "raw":
            x = self.model(x)
        elif self.hparams.input_type == "feature":
            x = self.model_h(x)
        return x

    def training_step(self, batch, batch_idx):
        loss = self.default_step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            outputs = self.default_step(batch, mode=f"valid_valid")
            self.validation0_step_outputs.append(outputs)
        elif dataloader_idx == 1:
            outputs = self.default_step(batch, mode=f"valid_test")
            self.validation1_step_outputs.append(outputs)

    def test_step(self, batch, batch_idx):
        outputs = self.default_step(batch, mode=f"test")
        self.test_step_outputs.append(outputs)

    def default_step(self, batch, mode):
        x, y, g, i = batch

        y_hat = self(x)
        pos_weight = torch.tensor([self.hparams.pos_weight], device=x.device)
        loss = self.criterion_fn(y_hat, y, reduction="mean", pos_weight=pos_weight)

        acc = self.metric.get_accuracy(y_hat, y, criterion=self.hparams.criterion)
        self.log_dict(
            {f"{mode}_loss": loss, f"{mode}_acc": acc},
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if len(batch) == 4 and mode != "train":
            return {"y_hat": y_hat.squeeze().detach(), "y": y, "g": g}

        return {"loss": loss, "y_hat": y_hat.squeeze().detach(), "y": y, "g": g}

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer: typing.Union[
            torch.optim.Optimizer, typing.Dict[str, torch.optim.Optimizer]
        ],
        optimizer_closure,
    ):
        # Please refer the following links for more information about optimizer_step
        # https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html

        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.learning_rate

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def on_train_epoch_end(self):
        if self.automatic_optimization == False:
            try:
                for sch in self.lr_schedulers():
                    sch.step()
            except:
                sch = self.lr_schedulers()
                sch.step()

    def eval_epoch_end(self, outputs, mode):
        """
        outputs: list (batch) of dictionaries (yhat, y, g), each dict value contains a list of tensors

        outputs[0]['y_hat']: y_hat tensor for each batch is saved as list
        outputs[0]['group']: group tensor for each batch is saved as list
        """
        acc_dict = {}
        len_dict = {}  # to calculate weighted average

        y_hat = torch.cat([elem["y_hat"] for elem in outputs])
        y = torch.cat([elem["y"] for elem in outputs])
        g = torch.cat([elem["g"] for elem in outputs])

        for y_idx in range(self.hparams.num_classes):
            for a_idx in range(self.hparams.num_attributes):
                mask = (y == y_idx) & (g == a_idx)
                if mask.sum() == 0:
                    acc_dict[(y_idx, a_idx)] = 0
                    len_dict[(y_idx, a_idx)] = 0
                    continue

                if y_idx == 0:
                    # threshold: 0 if no sigmoid else 1
                    acc = (y_hat[mask] < 0).sum() / mask.sum()
                else:
                    acc = (y_hat[mask] > 0).sum() / mask.sum()

                acc_dict[(y_idx, a_idx)] = acc
                len_dict[(y_idx, a_idx)] = mask.sum()

                self.log(
                    f"{mode}_acc_y_{y_idx}_a_{a_idx}",
                    acc,
                    prog_bar=True,
                    sync_dist=False,
                    add_dataloader_idx=False,
                )

        # worst acc should collect the minimun value among the non-zero length
        worst_acc = min(
            [acc for acc, len in zip(acc_dict.values(), len_dict.values()) if len != 0]
        )

        self.log_dict({f"{mode}_worst_acc": worst_acc})

        class_acc_list = []
        for y_idx in range(self.hparams.num_classes):
            acc = 0
            for a_idx in range(self.hparams.num_attributes):
                acc += acc_dict[(y_idx, a_idx)] * len_dict[(y_idx, a_idx)]
            acc /= max(
                sum(
                    [
                        len_dict[(y_idx, a_idx)]
                        for a_idx in range(self.hparams.num_attributes)
                    ]
                ),
                1,
            )
            class_acc_list.append(acc)
            self.log(
                f"{mode}_class_acc_y_{y_idx}",
                acc,
                prog_bar=True,
                sync_dist=False,
                add_dataloader_idx=False,
            )
        wc_acc = min(class_acc_list)
        self.log_dict({f"{mode}_worst_class_acc": wc_acc})

        # Weighted average of the accuracies
        avg_acc = sum(
            [acc * len for acc, len in zip(acc_dict.values(), len_dict.values())]
        ) / sum(len_dict.values())
        self.log_dict({f"{mode}_avg_acc": avg_acc})

        if mode == "valid_valid":
            # Worst group accuracy
            if worst_acc >= self.best_worst_acc and self.current_epoch > 0:
                self.best_epoch = self.current_epoch
                self.best_worst_acc = worst_acc
            self.log(
                f"{mode}_best_epoch", torch.tensor(self.best_epoch).to(torch.float32)
            )
            self.log(f"{mode}_best_worst_acc", self.best_worst_acc)

            # Worst class acc
            if wc_acc >= self.best_wc_acc and self.current_epoch > 0:
                self.best_wc_epoch = self.current_epoch
                self.best_wc_acc = wc_acc
            self.log(
                f"{mode}_best_wc_epoch",
                torch.tensor(self.best_wc_epoch).to(torch.float32),
            )
            self.log(f"{mode}_best_wc_acc", self.best_wc_acc)

        if mode == "valid_test" and self.best_epoch == self.current_epoch:
            self.log(f"{mode}_worst_acc_by_best_val_worst", worst_acc)
            self.log(f"{mode}_avg_acc_by_best_val_worst", avg_acc)
            for i in range(len(acc_dict.values())):
                self.log(
                    f"{mode}_group_{i}_acc_by_best_val_worst",
                    list(acc_dict.values())[i],
                )

        if mode == "valid_test" and self.best_wc_epoch == self.current_epoch:
            self.log(f"{mode}_worst_acc_by_best_val_cls_worst", worst_acc)
            self.log(f"{mode}_avg_acc_by_best_val_cls_worst", avg_acc)
            for i in range(len(acc_dict.values())):
                self.log(
                    f"{mode}_group_{i}_acc_by_best_val_cls_worst",
                    list(acc_dict.values())[i],
                )

    def on_validation_epoch_end(self):
        self.eval_epoch_end(self.validation0_step_outputs, "valid_valid")
        self.eval_epoch_end(self.validation1_step_outputs, "valid_test")
        self.validation0_step_outputs = []
        self.validation1_step_outputs = []

    def on_test_epoch_end(self):
        self.eval_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs = []

    def configure_optimizers(self):
        assert self.hparams.optimizer in [
            "sgd",
            "adam",
            "adamw",
        ], "Wrong optimizer name"

        if self.hparams.optimizer == "sgd":
            optim = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )

        elif self.hparams.optimizer == "adam":
            optim = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )

        elif self.hparams.optimizer == "adamw":
            optim = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )

        if self.hparams.scheduler == "multistep":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=optim, milestones=self.hparams.milestones, gamma=0.1
                ),
                "name": "lr_history",
            }
        elif self.hparams.scheduler == "cosineannealing":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optim, T_max=self.hparams.max_epochs
                ),
                "name": "lr_history",
            }
        else:
            scheduler = None

        return [optim], [scheduler]

    def get_criterion_fn(self, criterion):
        if criterion == "bce":

            def bce_loss(x: torch.Tensor, y: torch.Tensor, **kwargs):
                return F.binary_cross_entropy_with_logits(
                    x.reshape(-1), y.float().reshape(-1), **kwargs
                )

            criterion_fn = bce_loss
        elif criterion == "ce":
            criterion_fn = F.cross_entropy
        return criterion_fn
