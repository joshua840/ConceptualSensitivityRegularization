import os
import torch
from csr.module.dataset.feature_data_module import EpochChangeableFeatureDataset
from csr.module.utils.cav import compute_cav
from csr.module.models.load_model import load_model_head


# argument
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="catdog")
parser.add_argument("--model_name", type=str, default="convnext_t")
parser.add_argument("--cav_type", type=str, default="svm")
parser.add_argument("--root", type=str, default="/home/data/Features")
parser.add_argument("--save_path", type=str, default="/home/data/Features")
args = parser.parse_args()

dataset = EpochChangeableFeatureDataset(
    split="tr", root=args.root, dataset=args.dataset, model_name=args.model_name
)

w = compute_cav(
    vecs=dataset.x.numpy(), targets=dataset.y.numpy(), cav_type=args.cav_type
)

model = load_model_head(
    model="linear",
    activation_fn=None,
    softplus_beta=None,
    num_classes=1,
    ckpt_path=None,
    in_features=768,
    freeze=False,
    name="model_g",
)

model[1].load_state_dict({"weight": w, "bias": torch.tensor([0.0])})
from collections import OrderedDict

new_stdt = OrderedDict()

for k, v in model.state_dict().items():
    new_stdt["model_g." + k] = v


os.makedirs(args.save_path, exist_ok=True)
results = {"state_dict": new_stdt}
torch.save(
    results,
    os.path.join(
        args.save_path, f"{args.dataset}_{args.model_name}_{args.cav_type}.pt"
    ),
)
