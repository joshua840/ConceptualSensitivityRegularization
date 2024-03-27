from pytorch_lightning.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.max_epochs", "model.init_args.max_epochs")
        parser.link_arguments("seed_everything", "model.init_args.data_seed")
        parser.link_arguments("model.init_args.input_type", "model.init_args.data_type")


def cli_main():
    cli = MyLightningCLI(run=False)
    cli.trainer.fit(cli.model)

    # cli.trainer.fit(cli.model)
    # LitFeatureJTTMetadataGenerator

    # model = JTTMetadataGenerator(**vars(args))
    # model.prepare_data()
    # model.setup()
    # train_loader = model.train_dataloader()
    # train_loader2 = model.jtt_generation_dataloader()
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=train_loader2)


if __name__ == "__main__":
    cli_main()
