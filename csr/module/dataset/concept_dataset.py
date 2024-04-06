import os
import yaml
from PIL import Image
from torch.utils.data import Dataset


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
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.nimg_per_concept = nimg_per_concept
        self.dataset = dataset
        self.samples = self._yaml_to_img_target(dataset, nimg_per_concept)

    def _yaml_to_img_target(self, dataset, nimg_per_concept=50):
        # yaml file contains a dictionary with keys as concepts and values as list of image paths
        # this function convert it to a list of tuples (image_path, target)
        if dataset in ["waterbirds_concepts", "catdog_concepts"]:
            with open(f"configs/dataset/waterbirds_concepts.yaml", "r") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        elif dataset in ["celeba_cocnepts"]:
            with open(f"configs/dataset/celeba_concepts.yaml", "r") as f:
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

        if self.dataset in ["waterbirds_concepts", "catdog_concepts"]:
            sample = Image.open(os.path.join(self.root, "data_large", path)).convert(
                "RGB"
            )
        elif self.dataset in ["celeba_concepts"]:
            sample = Image.open(
                os.path.join(self.root, "img_align_celeba", path)
            ).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, 0, 0
