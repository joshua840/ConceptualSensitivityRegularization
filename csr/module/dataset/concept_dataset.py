import os
import yaml
from PIL import Image
from torch.utils.data import Dataset
import json


class ConceptDataset(Dataset):
    def __init__(
        self,
        root,
        dataset,
        model_name,
        nimg_per_concept=50,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        dataset = "waterbirds_concepts" if dataset == "catdog_concepts" else dataset
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.nimg_per_concept = nimg_per_concept
        self.dataset = dataset
        self.samples = self._yaml_to_img_target(dataset, nimg_per_concept)

    def _yaml_to_img_target(self, dataset, nimg_per_concept=50):
        # yaml file contains a dictionary with keys as concepts and values as list of image paths
        # this function convert it to a list of tuples (image_path, target)
        if dataset in ["celeba_collar_concepts"]:
            with open(f"configs/dataset/artifacts_celeba.json", "r") as f:
                data = json.load(f)
        elif dataset in [
            "celeba_collar_concepts_v2",
            "waterbirds_concepts_v2",
            "catdog_concepts_v2",
            "waterbirds_concepts",
            "catdog_concepts",
        ]:
            with open(f"configs/dataset/{dataset}.yaml", "r") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise NotImplementedError("dataset is not implemented")

        samples = []
        for i, (img_paths) in enumerate(data.values()):
            samples += [(img_path, i) for img_path in img_paths[:nimg_per_concept]]

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]

        middle_dir = {
            "waterbirds_concepts": "data_large",
            "catdog_concepts": "data_large",
            "celeba_collar_concepts": "img_align_celeba",
            "celeba_collar_concepts_v2": "img_align_celeba",
            "waterbirds_concepts_v2": "waterbird_complete95_forest2water2",
            "catdog_concepts_v2": "",
        }[self.dataset]

        sample = Image.open(os.path.join(self.root, middle_dir, path)).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, 0, 0
