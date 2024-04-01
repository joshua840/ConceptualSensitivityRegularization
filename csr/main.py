from lightning.pytorch.cli import LightningCLI
from csr.module.utils.save_config_callback import SaveConfigCallback


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.max_epochs", "model.init_args.max_epochs")
        parser.link_arguments("seed_everything", "model.init_args.data_seed")
        parser.link_arguments("model.init_args.input_type", "model.init_args.data_type")


def cli_main():
    cli = MyLightningCLI(run=False, save_config_callback=SaveConfigCallback)

    cli.trainer.fit(cli.model)
    cli.trainer.test(cli.model, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
