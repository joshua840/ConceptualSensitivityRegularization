import os
import yaml
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ConceptDataset(Dataset):
    def __init__(
        self,
        root,
        dataset,
        nimg_per_concept=50,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self.__yaml_to_img_target__(dataset, nimg_per_concept)

    def __yaml_to_img_target__(self, dataset, nimg_per_concept=50):
        # yaml file contains a dictionary with keys as concepts and values as list of image paths
        # this function convert it to a list of tuples (image_path, target)
        if dataset in ["waterbirds_concepts", "catdog_concepts"]:
            with open(f"configs/dataset/{dataset}.yaml", "r") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise NotImplementedError("dataset is not implemented")

        samples = []
        for concept, img_paths in data.items():
            for img_path in img_paths[:nimg_per_concept]:
                samples.append((img_path, concept))
        return samples

    def __len__(self):
        return len(self.nimg_per_concept) * len(self.img_labels.keys())

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = Image.open(os.path.join(self.root, "data_large", path)).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, 0, 0
