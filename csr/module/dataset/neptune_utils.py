import os
import torch
from collections import OrderedDict


def load_ckpt(exp_id, root_dir):
    filename = [i for i in os.listdir(os.path.join(root_dir, exp_id)) if "last.ckpt" not in i]
    path = os.path.join(root_dir, exp_id, filename[0])
    return torch.load(path, map_location="cpu")


def wandb_load_ckpt(exp_id, root_dir):
    path = os.path.join(root_dir, exp_id, "checkpoints/last.ckpt")
    return torch.load(path, map_location="cpu")


def set_prev_args(ckpt, args):
    for k, v in ckpt["hyper_parameters"].items():
        if k == "data_dir":
            continue
        if k == "exp_id":
            continue
        if k == "default_root_dir":
            continue
        if k == "mratio":
            v = 1.0
        setattr(args, k, v)
    return args


def safe_model_loader(model, ckpt):
    try:
        model.load_state_dict(ckpt["state_dict"])
    except:
        new_state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            if "model_g" in k or "linear_model" in k:
                continue
            new_state_dict[k.removeprefix("model.")] = v

        model.model.load_state_dict(new_state_dict)
    return
