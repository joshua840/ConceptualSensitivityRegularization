import torch
from .models.load_model import load_model_head
from . import CGR


class MultiCGR(CGR):
    def __init__(
        self,
        g_ckpt_path: list = [],
        g_num_heads: int = 1,
        **kwargs,
    ):
        """
        Args:
            g_ckpt_path: str, ckpt path for model_g if needed
            g_num_heads: int, number of heads of model_g
        """
        assert len(g_ckpt_path) == g_num_heads, "len(g_ckpt_path) != g_num_heads"
        assert self.hparams.cgr_stage == "stage2", "CGR stage1 is not supported"

        super().__init__(**kwargs)
        kwargs["module_name"] = "MultiCGR"
        self.save_hyperparameters()

        self.model_g = torch.nn.ModuleList(
            [
                load_model_head(
                    model=self.hparams.g_model,
                    activation_fn=self.hparams.g_activation,
                    softplus_beta=self.hparams.g_softplus_beta,
                    num_classes=self.hparams.g_num_classes,
                    ckpt_path=path,
                    in_features=self.in_features,
                    freeze=self.hparams.g_freeze,
                )
                for path in g_ckpt_path
            ]
        )

    def classifier_step(self, data, mode):
        x, y, g, i = data

        x.requires_grad = True if mode == "train" else False

        y_hat = self(x)
        pos_weight = torch.tensor([self.hparams.pos_weight], device=x.device)
        ce_loss = self.criterion_fn(y_hat, y, reduction="mean", pos_weight=pos_weight)
        loss = ce_loss

        g_logit_list = [g_head(x) for g_head in self.model_g]

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

                grad_g_list = []
                for g_head in self.model_g:
                    grad_g = torch.autograd.grad(
                        outputs=g_head(x).sum(),
                        inputs=x,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    grad_g_list.append(grad_g)
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

                grad_g_list = []
                for g_head in self.model_g:
                    cav_ce_loss = self.g_criterion_fn(g_head(x), y, reduction="mean")
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
                    grad_g_list.append(grad_g)

            if "cross" in self.hparams.cs_method:
                # concept_gradient = grad_h @ grad_g.T
                concept_gradient_list = [grad_h @ grad_g.T for grad_g in grad_g_list]
                feature_distance = (x @ x.T + 1) / 2  # [0, 1]

                if self.hparams.cs_method == "cross_dot_sq":
                    # cs_loss = (concept_gradient.square() * feature_distance).mean()
                    cs_loss = sum(
                        [
                            (concept_gradient.square() * feature_distance).mean()
                            for concept_gradient in concept_gradient_list
                        ]
                    )

            else:
                # concept_gradient = torch.einsum("nc,nc->n", grad_h, grad_g)
                concept_gradient_list = [
                    torch.einsum("nc,nc->n", grad_h, grad_g) for grad_g in grad_g_list
                ]

                if self.hparams.cs_method == "dot_sq":
                    # cs_loss = concept_gradient.square().mean()
                    cs_loss = sum(
                        [
                            concept_gradient.square().mean()
                            for concept_gradient in concept_gradient_list
                        ]
                    )
                elif self.hparams.cs_method == "dot_abs":
                    # cs_loss = concept_gradient.abs().mean()
                    cs_loss = sum(
                        [
                            concept_gradient.abs().mean()
                            for concept_gradient in concept_gradient_list
                        ]
                    )
                else:
                    # concept_gradient = concept_gradient / (
                    #     (
                    #         grad_g.norm(dim=1, keepdim=True)
                    #         * grad_h.norm(dim=1, keepdim=True)
                    #     )
                    #     + 1e-7
                    # )
                    concept_gradient_list = [
                        concept_gradient
                        / (
                            (
                                grad_g.norm(dim=1, keepdim=True)
                                * grad_h.norm(dim=1, keepdim=True)
                            )
                            + 1e-7
                        )
                        for concept_gradient, grad_g in zip(
                            concept_gradient_list, grad_g_list
                        )
                    ]
                    if self.hparams.cs_method == "cosine_sq":
                        # cs_loss = concept_gradient.square().mean()
                        cs_loss = sum(
                            [
                                concept_gradient.square().mean()
                                for concept_gradient in concept_gradient_list
                            ]
                        )
                    elif self.hparams.cs_method == "cosine_abs":
                        # cs_loss = concept_gradient.abs().mean()
                        cs_loss = sum(
                            [
                                concept_gradient.abs().mean()
                                for concept_gradient in concept_gradient_list
                            ]
                        )
                    elif self.hparams.cs_method == "cosine_log":
                        # cs_loss = torch.log((1 - concept_gradient.abs()) + 1e-7).mean()
                        cs_loss = sum(
                            [
                                torch.log((1 - concept_gradient.abs()) + 1e-7).mean()
                                for concept_gradient in concept_gradient_list
                            ]
                        )

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