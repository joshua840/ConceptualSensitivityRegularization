import os
import torch
from csr.module.dataset.feature_data_module import EpochChangeableFeatureDataset
from csr.module.utils.cav import compute_cav

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

w = compute_cav(vecs=dataset.x, targets=dataset.y, cav_type=args.cav_type)

# save
torch.save(
    w,
    os.path.join(
        args.save_path, f"{args.dataset}_{args.model_name}_{args.cav_type}.pt"
    ),
)
