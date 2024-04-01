import os
import torch
import lightning.pytorch as pl
import pandas as pd
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
import pandas as pd
from .common_spurious_dataset import CommonSpuriousDataset


class ColoredMNIST(CommonSpuriousDataset):
    def __init__(
        self,
        data_dir,
        split,
        data_seed=1234,
        minor_ratio=0.1,
        subsample_what=None,
        upsample_indices=None,
    ):
        fname = f"{split}_seed_{data_seed}.pt"
        data_path = os.path.join(data_dir, "ColoredMNIST", fname)

        # check if the data is already generated
        print(data_path)
        if os.path.exists(data_path):
            self.i, self.x, self.y, self.g = torch.load(data_path)
        else:
            raise NotImplementedError()

        # minor_ratio
        if split == "tr" and minor_ratio is not None:
            indices_dict = self._get_indices_dict()
            indices_dict = self._set_minor_ratio(indices_dict, minor_ratio)
            self.i = np.concatenate(
                [indices for indices in indices_dict.values()], axis=0
            )

        if subsample_what is not None:
            indices_dict = self._subsample_group(indices_dict)
            self.i = np.concatenate(
                [indices for indices in indices_dict.values()], axis=0
            )

        if upsample_indices is not None:
            self.i = np.concatenate([self.i, np.array(upsample_indices)], axis=0)

        self.count_groups()

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

    def transform(self, x):
        return x

    def __getitem__(self, i):
        j = self.i[i]
        x = self.transform(self.x[j])
        y = self.y[j]
        g = self.g[j]
        return x, y, g, j

    def __len__(self):
        return len(self.i)

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def num_groups(self) -> int:
        return 4

    # def get_input_features(self):
    # return (14, 14, 2)

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

    def _set_minor_ratio(self, indices_selector_dict, minor_ratio):
        new_length = int(len(indices_selector_dict[(1, 1)]) * minor_ratio)
        indices_selector_dict[(0, 1)] = indices_selector_dict[(0, 1)][:new_length]

        new_length = int(len(indices_selector_dict[(0, 0)]) * minor_ratio)
        indices_selector_dict[(1, 0)] = indices_selector_dict[(1, 0)][:new_length]
        return indices_selector_dict

    def _subsample_group(self, indices_selector_dict):
        min_length = min([len(indices) for indices in indices_selector_dict.values()])
        for key in indices_selector_dict.keys():
            indices_selector_dict[key] = indices_selector_dict[key][:min_length]
        return indices_selector_dict

    @staticmethod
    def make_dataset(data_dir, seed):
        pl.seed_everything(seed)

        mnist = torchvision.datasets.MNIST(data_dir, train=True, download=True)
        mnist_train = (mnist.data[:50000], mnist.targets[:50000])
        mnist_val = (mnist.data[50000:], mnist.targets[50000:])

        mnist = torchvision.datasets.MNIST(data_dir, train=False, download=True)
        mnist_test = (mnist.data, mnist.targets)

        rng_state = np.random.get_state()
        np.random.shuffle(mnist_train[0].numpy())
        np.random.set_state(rng_state)
        np.random.shuffle(mnist_train[1].numpy())

        # Build environments
        def make_environment(images, labels, e):
            def torch_bernoulli(p, size):
                return (torch.rand(size) < p).long()

            def torch_xor(a, b):
                return (a - b).abs()  # Assumes both inputs are either 0 or 1

            # 2x subsample for computational convenience
            images = images.reshape((-1, 28, 28))
            # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
            # Assign a binary label based on the digit; flip label with probability 0.25
            labels = (labels < 5).long()
            labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
            # Assign a color based on the label; flip the color with probability e
            colors = torch_xor(labels, torch_bernoulli(e, len(labels)))

            # Apply the color to the image by zeroing out the other color channel
            images = torch.stack([images, images, images], dim=1)
            images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
            images[torch.tensor(range(len(images))), 2, :, :] *= 0
            return (
                list(range(len(labels))),
                (images.float() / 255.0),
                labels.long(),
                colors.long(),
            )

        tr = make_environment(mnist_train[0], mnist_train[1], 0.1)
        va = make_environment(mnist_val[0], mnist_val[1], 0.1)
        te = make_environment(mnist_test[0], mnist_test[1], 0.5)

        # create the directory if it does not exist

        fname = f"tr_seed_{seed}.pt"
        data_path = os.path.join(data_dir, "ColoredMNIST", fname)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save(tr, data_path)

        fname = f"va_seed_{seed}.pt"
        data_path = os.path.join(data_dir, "ColoredMNIST", fname)
        torch.save(va, data_path)

        fname = f"te_seed_{seed}.pt"
        data_path = os.path.join(data_dir, "ColoredMNIST", fname)
        torch.save(te, data_path)

        return


class ColoredDummy(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        data_seed,
        minor_ratio,
        split,
        subsample_what=None,
        upsample_count=None,
        upsample_indices_path=None,
    ):
        fname = f"{split}_seed{data_seed}.pt"
        data_path = os.path.join(data_root, "ColoredDummy", fname)

        # check if the data is already generated
        if os.path.exists(data_path):
            self.index, self.x, self.y, self.g = torch.load(data_path)
        else:
            raise NotImplementedError()

    def transform(self, x):
        return x

    def __getitem__(self, i):
        j = self.index[i]
        x = self.transform(self.x[j])
        y = self.y[j]
        g = self.g[j]
        return x, y, g

    def __len__(self):
        return len(self.index)

    @staticmethod
    def make_dataset(data_root, seed):
        pl.seed_everything(seed)

        mnist = torchvision.datasets.MNIST(
            "~/datasets/mnist", train=True, download=True
        )
        x = mnist.data
        np.random.shuffle(x.numpy())
        xs = [x[i : i + 6000] for i in range(0, len(x), 6000)]
        x = sum(xs)

        x_tr = x[:5000]
        x_val = x[5000:]

        def make_environment(images):
            # 2x subsample for computational convenience
            images = images.reshape((-1, 28, 28))[:, ::2, ::2]
            colors = torch.randint(low=0, high=2, size=(len(images),))

            # Apply the color to the image by zeroing out the other color channel
            images = torch.stack([images, images], dim=1)
            images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
            return (
                list(range(len(colors))),
                (images.float() / 255.0),
                colors.long(),
                colors.long(),
            )

        tr = make_environment(x_tr)
        val = make_environment(x_val)

        # create the directory if it does not exist
        fname = f"tr_seed{seed}.pt"
        data_path = os.path.join(data_root, "ColoredDummy", fname)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save(tr, data_path)

        fname = f"val_seed{seed}.pt"
        data_path = os.path.join(data_root, "ColoredDummy", fname)
        torch.save(val, data_path)

        return
