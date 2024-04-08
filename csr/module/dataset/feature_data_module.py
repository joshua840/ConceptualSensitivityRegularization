from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import os

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from torch.utils.data import TensorDataset, DataLoader
from .common_spurious_dataset import CommonSpuriousDataset
from . import CelebA, Waterbirds, ColoredMNIST, Dogs, CatDog, CelebACollar


class EpochChangeableFeatureDataset(CommonSpuriousDataset):
    def __init__(
        self,
        split,
        root,
        dataset,
        model_name,
        minor_ratio=None,
        subsample_what=None,
        upsample_indices=None,
    ):
        """
        split : train, val, test
        Dataset/tr/epoch0~599/data.npy
        or
        Features/Dataset/MODEL_NAME/tr/{0...599}.pt
        Features/Dataset/MODEL_NAME/tr/metadata.pt
        Features/Dataset/MODEL_NAME/va/0.pt
        Features/Dataset/MODEL_NAME/va/metadata.pt
        Features/Dataset/MODEL_NAME/te/0.pt
        Features/Dataset/MODEL_NAME/te/metadata.pt
        These features are saved without shuffling
        """
        self.split = split
        self.dataset_name = dataset
        self.data_root = os.path.join(root, "Features", dataset, model_name, split)

        # find the maximum epochs in the tr directory
        max_epochs = max(
            [
                int(i.split(".")[0])
                for i in os.listdir(self.data_root)
                if i != "metadata.pt"
            ]
        )

        # init shuffle indices
        self.shuffle_indices = list(range(max_epochs + 1))
        np.random.shuffle(self.shuffle_indices)

        self.load_new_x(current_epoch=0)
        self.y, self.g, self.i = torch.load(os.path.join(self.data_root, "metadata.pt"))

        # minor_ratio
        if split == "tr" and minor_ratio is not None:
            indices_dict = self._get_indices_dict()
            indices_dict = self._remove_label_bias(indices_dict)
            indices_dict = self._set_minor_ratio(indices_dict, minor_ratio)
            self.i = np.concatenate(
                [indices for indices in indices_dict.values()], axis=0
            )

        self.count_groups()
        print(split, self.group_sizes)

        if subsample_what is not None:
            indices_dict = self._subsample_group(indices_dict)
            self.i = np.concatenate(
                [indices for indices in indices_dict.values()], axis=0
            )
            self.count_groups()
            print(split, "after subsample", self.group_sizes)

        # for JTT
        if upsample_indices is not None:
            self.i = np.concatenate([self.i, np.array(upsample_indices)], axis=0)

            print(self.i)
            self.count_groups()
            print(split, "after upsample", self.group_sizes)

    def count_groups(self):
        self.wg, self.wy = [], []

        self.nb_groups = len(set(self.g.tolist()))
        self.nb_labels = len(set(self.y.tolist()))
        self.group_sizes = {}  # [0] * self.nb_groups * self.nb_labels
        self.class_sizes = [0] * self.nb_labels

        for g in range(self.nb_groups):
            for y in range(self.nb_labels):
                self.group_sizes[(g, y)] = (
                    (self.y[self.i] == y) * (self.g[self.i] == g)
                ).sum()
                self.class_sizes.append((self.y[self.i] == y).sum())

    def _get_indices_dict(self):
        indices_dict = {}
        for attr in range(2):
            for label in range(2):
                indices_dict[(attr, label)] = (
                    ((self.g == attr) * (self.y == label)).nonzero().squeeze()
                )
        return indices_dict

    def _remove_label_bias(self, indices_dict):
        return indices_dict

    # TODO: fix this! This code should be different for each dataset
    def _set_minor_ratio(self, indices_selector_dict, minor_ratio):
        if "concepts" in self.dataset_name:
            return indices_selector_dict
        set_minor_ratio = {
            "colored_mnist": ColoredMNIST._set_minor_ratio,
            "celeba": CelebA._set_minor_ratio,
            "celeba_collar": CelebACollar._set_minor_ratio,
            "celeba_gender": CelebA._set_minor_ratio,
            "waterbirds": Waterbirds._set_minor_ratio,
            "dogs": Dogs._set_minor_ratio,
            "catdog": CatDog._set_minor_ratio,
        }[self.dataset_name]
        return set_minor_ratio(indices_selector_dict, minor_ratio)

    def _subsample_group(self, indices_selector_dict):
        min_length = min([len(indices) for indices in indices_selector_dict.values()])
        for key in indices_selector_dict.keys():
            indices_selector_dict[key] = indices_selector_dict[key][:min_length]
        return indices_selector_dict

    def __getitem__(self, i):
        j = self.i[i]
        x = self.x[j].float()
        y = self.y[j]
        g = self.g[j]
        return (x, y, g, torch.tensor(i, dtype=torch.long))

    def __len__(self):
        return len(self.i)

    def _group_counts(self):
        # TODO: Meaningless code
        group = self.g
        _, counts = np.unique(np.array(group), return_counts=True)
        return torch.tensor(counts)

    def load_new_x(self, current_epoch):
        epoch = self.shuffle_indices[current_epoch]
        self.x = torch.load(os.path.join(self.data_root, f"{epoch}.pt"))

        if len(self.x[0].shape) != 1:
            self.x = torch.cat(self.x, dim=0)
        return


class EpochChangeableTensorDataset(TensorDataset):
    def __init__(
        self,
        tensors,
        minor_ratio=None,
        minimum_samples_per_group=0,
        subsample=None,
        split="full",
    ):
        """
        tensors: tensor data
        minor_ratio: ratio of minority samples
        subsample: subsample dataset [None, 'groups', 'labels']
        """
        super().__init__(*tensors)
        self.current_epoch = 0
        self.minor_ratio = minor_ratio
        self.indices = list(range(self.tensors[0][0].size(0)))
        self.subsample = subsample
        self.split = split
        self.minimum_samples_per_group = minimum_samples_per_group

        self.subsample_(minor_ratio, subsample, minimum_samples_per_group, split)

        self.vervose()

    def __getitem__(self, index):
        return tuple(
            tensor[self.current_epoch][self.indices[index]] for tensor in self.tensors
        )

    def __len__(self):
        return len(self.indices)

    def vervose(self):
        label = self.tensors[1][0][self.indices]
        attr = self.tensors[2][0][self.indices, 0]
        group = label * 2 + attr
        n_data_groups = [(group == idx).sum() for idx in range(4)]

        print(
            f"mr: {self.minor_ratio}\t{self.subsample}\t{self.split}\t\tgroup 0/1/2/3:\t{n_data_groups[0]}\t{n_data_groups[1]}\t{n_data_groups[2]}\t{n_data_groups[3]}\t||original tensor: {self.tensors[0].shape}"
        )

    def set_epochs(self, epoch):
        self.current_epoch = epoch

    def subsample_(self, minor_ratio, subsample, minimum_samples_per_group, split):
        label = self.tensors[1][0]
        attr = self.tensors[2][0][:, 0]
        group = 2 * label + attr

        # group indices without empty list
        temp = [list((group == idx).nonzero().flatten()) for idx in range(4)]
        group_indices_list = [lst for lst in temp if len(lst) != 0]

        # apply minor_ratio
        if minor_ratio != None:
            if len(group_indices_list) == 4:
                # major: 0 & 3 // minor: 1 & 2
                num_minor_label0 = max(
                    minimum_samples_per_group,
                    int(len(group_indices_list[0]) * minor_ratio),
                )
                num_minor_label1 = max(
                    minimum_samples_per_group,
                    int(len(group_indices_list[3]) * minor_ratio),
                )

                group_indices_list[1] = group_indices_list[1][:num_minor_label0]
                group_indices_list[2] = group_indices_list[2][:num_minor_label1]
            elif len(group_indices_list) == 2:
                assert len(group_indices_list[0]) == 466  # only for waterbirds dataset
                group_indices_list[0] = group_indices_list[0][
                    : max(minimum_samples_per_group, int(467 * minor_ratio))
                ]
                group_indices_list[1] = group_indices_list[1][
                    : max(minimum_samples_per_group, int(133 * minor_ratio))
                ]

        # split
        if split != "full":
            for idx, group_indices in enumerate(group_indices_list):
                length = len(group_indices)
                group_indices_list[idx] = (
                    group_indices[: length // 2]
                    if split == "front"
                    else group_indices[length // 2 :]
                )

        self.indices = []
        if subsample == "groups":
            # subsample
            n_sub = min([len(group_indices) for group_indices in group_indices_list])

            for group_indices in group_indices_list:
                self.indices += group_indices[:n_sub]
        else:
            for group_indices in group_indices_list:
                self.indices += group_indices

        self.indices = [int(i) for i in self.indices]


class EpochChangeableConceptDataset(TensorDataset):
    def __init__(self, tensors, nimg_per_concept=None):
        super().__init__(*tensors)
        self.current_epoch = 0

        self.setup(nimg_per_concept)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

    def setup(self, nimg_per_concept):
        # self.tuples = (x, y) // [epoch][datasamples][extra_dim]
        label = self.tensors[1][0]

        label0_indices = (
            (label == 0)
            .nonzero()
            .flatten()[: min(nimg_per_concept, (label == 0).sum())]
        )
        label1_indices = (
            (label == 1)
            .nonzero()
            .flatten()[: min(nimg_per_concept, (label == 1).sum())]
        )

        indices = list(label0_indices) + list(label1_indices)

        tensor1 = self.tensors[1][:, indices].flatten()
        tensor0 = self.tensors[0][:, indices].reshape(len(tensor1), -1)

        self.tensors = (tensor0, tensor1)


class SpuriousFeatureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        dataset: str,
        batch_size_train: int = 50,
        batch_size_test: int = 100,
        num_workers=8,
        minor_ratio=0.0,
        tr_shuffle=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        return

    def setup(self, stage=None):
        data = torch.load(self.hparams.data_path)

        self.dataset_train = EpochChangeableTensorDataset(
            (
                data["tr_spurious"]["fx"],
                data["tr_spurious"]["y"],
                data["tr_spurious"]["g"],
            ),
            self.hparams.minor_ratio,
        )

        self.dataset_major_val = EpochChangeableTensorDataset(
            (data["val_major"]["fx"], data["val_major"]["y"], data["val_major"]["g"])
        )
        self.dataset_minor_val = EpochChangeableTensorDataset(
            (data["val_major"]["fx"], data["val_minor"]["y"], data["val_minor"]["g"])
        )

        self.dataset_major_test = EpochChangeableTensorDataset(
            (data["te_major"]["fx"], data["te_major"]["y"], data["te_major"]["g"])
        )
        self.dataset_minor_test = EpochChangeableTensorDataset(
            (data["te_minor"]["fx"], data["te_minor"]["y"], data["te_minor"]["g"])
        )

    def get_val_dataloader(self):
        val_major_dataloader = DataLoader(
            self.dataset_major_val,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        val_minor_dataloader = DataLoader(
            self.dataset_minor_val,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        return val_major_dataloader, val_minor_dataloader

    def get_test_dataloader(self):
        test_major_dataloader = DataLoader(
            self.dataset_major_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        test_minor_dataloader = DataLoader(
            self.dataset_minor_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

        return test_major_dataloader, test_minor_dataloader

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size_train,
            shuffle=self.hparams.tr_shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        major_val_dataloader, minor_val_dataloader = self.get_val_dataloader()
        return [major_val_dataloader, minor_val_dataloader]

    def test_dataloader(self):
        major_test_dataloader, minor_test_dataloader = self.get_test_dataloader()
        return [major_test_dataloader, minor_test_dataloader]

    def predict_dataloader(self):
        return self.test_dataloader()

    def get_num_classes(self):
        num_cls_dict = {
            "SpuriousFlowers17": 10,
            "Spuriousoxford-iiit-pet": 37,
            "SpuriousCatDog": 2,
            "NonSpuriousCatDog": 2,
            "Waterbirds": 2,
            "dogs": 2,
        }

        return num_cls_dict[self.hparams.dataset]

    @classmethod
    def add_data_specific_args(
        cls, parent_parser: ArgumentParser, **kwargs
    ) -> ArgumentParser:
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        group = parser.add_argument_group("Data arguments")
        group.add_argument(
            "--data_seed", default=1234, type=int, help="batchsize of data loaders"
        )
        group.add_argument(
            "--num_workers", default=8, type=int, help="number of workers"
        )
        group.add_argument(
            "--batch_size_train", default=50, type=int, help="batchsize of data loaders"
        )
        group.add_argument(
            "--batch_size_test", default=100, type=int, help="batchsize of data loaders"
        )
        group.add_argument(
            "--data_path",
            default="~/Data",
            type=str,
            help="directory of feature dataset",
        )
        group.add_argument(
            "--minor_ratio",
            default=0.0,
            type=float,
            help="ratio of minor group in training dataset",
        )
        return parser


class ConceptFeatureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        dataset: str,
        batch_size_train: int = 50,
        batch_size_test: int = 100,
        nimg_per_concept=50,
        num_workers=8,
        tr_shuffle=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        return

    def setup(self, stage=None):
        data = torch.load(self.hparams.data_path)

        self.dataset_train = EpochChangeableConceptDataset(
            (data["tr_concept"]["fx"], data["tr_concept"]["y"]),
            self.hparams.nimg_per_concept,
        )
        self.dataset_val = EpochChangeableConceptDataset(
            (data["val_concept"]["fx"], data["val_concept"]["y"]),
            self.hparams.nimg_per_concept,
        )
        self.dataset_test = EpochChangeableConceptDataset(
            (data["te_concept"]["fx"], data["te_concept"]["y"]),
            self.hparams.nimg_per_concept,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size_train,
            shuffle=self.hparams.tr_shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    @classmethod
    def add_data_specific_args(
        cls, parent_parser: ArgumentParser, **kwargs
    ) -> ArgumentParser:
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        group = parser.add_argument_group("Data arguments")
        group.add_argument(
            "--data_seed", default=1234, type=int, help="batchsize of data loaders"
        )
        group.add_argument(
            "--num_workers", default=8, type=int, help="number of workers"
        )
        group.add_argument(
            "--batch_size_train", default=50, type=int, help="batchsize of data loaders"
        )
        group.add_argument(
            "--batch_size_test", default=100, type=int, help="batchsize of data loaders"
        )
        group.add_argument(
            "--nimg_per_concept",
            default=100,
            type=int,
            help="number of images for each concept",
        )
        group.add_argument(
            "--data_path",
            default="~/Data",
            type=str,
            help="directory of feature dataset",
        )
        return parser


class SpuriousConceptFeatureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        dataset: str,
        spurious_batch_size_train: int = 50,
        spurious_batch_size_test: int = 100,
        concept_batch_size_train: int = 50,
        concept_batch_size_test: int = 100,
        nimg_per_concept: int = 50,
        num_workers=8,
        minor_ratio=0.0,
        tr_shuffle=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        return

    def setup(self, stage=None):
        data = torch.load(self.hparams.data_path)

        self.dataset_train = EpochChangeableTensorDataset(
            (
                data["tr_spurious"]["fx"],
                data["tr_spurious"]["y"],
                data["tr_spurious"]["g"],
            ),
            self.hparams.minor_ratio,
        )

        self.dataset_major_val = EpochChangeableTensorDataset(
            (data["val_major"]["fx"], data["val_major"]["y"], data["val_major"]["g"])
        )

        self.dataset_minor_val = EpochChangeableTensorDataset(
            (data["val_major"]["fx"], data["val_minor"]["y"], data["val_minor"]["g"])
        )

        self.dataset_major_test = EpochChangeableTensorDataset(
            (data["te_major"]["fx"], data["te_major"]["y"], data["te_major"]["g"])
        )
        self.dataset_minor_test = EpochChangeableTensorDataset(
            (data["te_minor"]["fx"], data["te_minor"]["y"], data["te_minor"]["g"])
        )

        self.dataset_concept_train = EpochChangeableConceptDataset(
            (data["tr_concept"]["fx"], data["tr_concept"]["y"]),
            self.hparams.nimg_per_concept,
        )
        self.dataset_concept_val = EpochChangeableConceptDataset(
            (data["val_concept"]["fx"], data["val_concept"]["y"]),
            self.hparams.nimg_per_concept,
        )
        self.dataset_concept_test = EpochChangeableConceptDataset(
            (data["te_concept"]["fx"], data["te_concept"]["y"]),
            self.hparams.nimg_per_concept,
        )

    def get_val_dataloader(self):
        val_major_dataloader = DataLoader(
            self.dataset_major_val,
            batch_size=self.hparams.spurious_batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        val_minor_dataloader = DataLoader(
            self.dataset_minor_val,
            batch_size=self.hparams.spurious_batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        val_concept_dataloader = DataLoader(
            self.dataset_concept_val,
            batch_size=self.hparams.concept_batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        return val_major_dataloader, val_minor_dataloader, val_concept_dataloader

    def get_test_dataloader(self):
        test_major_dataloader = DataLoader(
            self.dataset_major_test,
            batch_size=self.hparams.spurious_batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        test_minor_dataloader = DataLoader(
            self.dataset_minor_test,
            batch_size=self.hparams.spurious_batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        test_concept_dataloader = DataLoader(
            self.dataset_concept_test,
            batch_size=self.hparams.concept_batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        return test_major_dataloader, test_minor_dataloader, test_concept_dataloader

    def train_dataloader(self):
        data_loader = DataLoader(
            self.dataset_train,
            batch_size=self.hparams.spurious_batch_size_train,
            shuffle=self.hparams.tr_shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        concept_loader = DataLoader(
            self.dataset_concept_train,
            batch_size=self.hparams.concept_batch_size_train,
            shuffle=self.hparams.tr_shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return {"classifier": data_loader, "concept": concept_loader}
        # loaders = {"classifier": data_loader, "concept": concept_loader}
        # return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        return self.get_val_dataloader()

    def test_dataloader(self):
        return self.get_test_dataloader()

    def predict_dataloader(self):
        return self.test_dataloader()

    def get_num_classes(self):
        num_cls_dict = {
            "SpuriousFlowers17": 10,
            "Spuriousoxford-iiit-pet": 37,
            "SpuriousCatDog": 2,
            "NonSpuriousCatDog": 2,
            "Waterbirds": 2,
        }

        return num_cls_dict[self.hparams.dataset]

    @classmethod
    def add_data_specific_args(
        cls, parent_parser: ArgumentParser, **kwargs
    ) -> ArgumentParser:
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        group = parser.add_argument_group("Data arguments")
        group.add_argument(
            "--data_seed", default=1234, type=int, help="batchsize of data loaders"
        )
        group.add_argument(
            "--num_workers", default=8, type=int, help="number of workers"
        )
        group.add_argument(
            "-sbatch_tr",
            "--spurious_batch_size_train",
            default=50,
            type=int,
            help="batchsize of data loaders",
        )
        group.add_argument(
            "-sbatch_te",
            "--spurious_batch_size_test",
            default=100,
            type=int,
            help="batchsize of data loaders",
        )
        group.add_argument(
            "-cbatch_tr",
            "--concept_batch_size_train",
            default=50,
            type=int,
            help="batchsize of data loaders",
        )
        group.add_argument(
            "-cbatch_te",
            "--concept_batch_size_test",
            default=100,
            type=int,
            help="batchsize of data loaders",
        )
        group.add_argument(
            "--nimg_per_concept",
            default=100,
            type=int,
            help="number of images for each concept",
        )
        group.add_argument("--minor_ratio", default=0, type=float, help="minor ratio")
        group.add_argument(
            "--data_path",
            default="~/Data",
            type=str,
            help="directory of feature dataset",
        )
        return parser
