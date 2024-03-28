import torch
from pytorch_lightning.cli import LightningCLI

from tqdm import tqdm
import os


class Fhook:
    def __init__(self):
        pass

    def save_outputs_hook(self):
        def fn(_, __, output):
            self._hooked_features = output.detach().cpu()

        return fn


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--save_root", type=str)
        parser.add_argument("--target_layer", type=str)
        parser.add_argument("--dataset", type=str)
        parser.link_arguments("model.init_args.dataset", "dataset")


def cli_main():
    cli = MyLightningCLI(run=False, subclass_mode_data=True)
    # save_config_kwargs={"overwrite": True}
    args = cli.config

    # data_module = cli.datamodule
    data_module = cli.model
    data_module.prepare_data()

    # setattr(args, "num_classes", data_module.num_classes)
    model = cli.model
    model = model.cuda()

    tr_loader = data_module.generation_dataloader()
    va_loader = data_module.val_dataloader()
    te_loader = data_module.test_dataloader()

    loaders_dict = {
        "tr": tr_loader,
        "va": va_loader[0],
        "te": te_loader,
    }

    with torch.no_grad():
        for key, loader in loaders_dict.items():
            print(key)
            hook = Fhook()
            for name, module in model.named_modules():
                if name == "model." + args.target_layer:
                    handle = module.register_forward_hook(hook.save_outputs_hook())
                    print("hook is succesfully registered")

            if "tr" in key:
                epochs = cli.config.trainer.max_epochs
                model = model.train()
            else:
                epochs = 1
                model = model.eval()

            for epoch in range(epochs):
                fs = []
                if epoch == 0:
                    ys = []
                    gs = []

                for data in tqdm(loader, desc=f"Epoch {epoch}"):
                    x, y, g, i = data

                    x = x.cuda()
                    model(x)

                    fs.append(
                        torch.flatten(hook._hooked_features, start_dim=1).detach().cpu()
                    )

                    if epoch == 0:
                        ys.append(y.detach().cpu())
                        gs.append(g.detach().cpu())
                fs = torch.cat(fs, dim=0).half()

                os.makedirs(
                    f"{args.save_root}/Features/{args.dataset}/{key}", exist_ok=True
                )
                torch.save(
                    fs, f"{args.save_root}/Features/{args.dataset}/{key}/{epoch}.pt"
                )

                if epoch == 0:
                    ys = torch.cat(ys, dim=0)
                    gs = torch.cat(gs, dim=0)
                    torch.save(
                        [ys, gs, range(len(gs))],
                        f"{args.save_root}/Features/{args.dataset}/{key}/metadata.pt",
                    )

    handle.remove()


if __name__ == "__main__":
    cli_main()
