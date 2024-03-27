import torch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from module.dataset.data_module import SpuriousConceptDataModule
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sns.set(style="white", context="notebook", rc={"figure.figsize": (14, 10)})
from module.pl_spurious_trainer import LitSpuriousClassifier
from module.pl_spurious_csr_trainer import LitSpuriousCSRClassifier
from module.pl_spurious_cgr_trainer import LitSpuriousCGRClassifier
from module.pl_spurious_lnl_trainer import LitSpuriousLNLClassifier

parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--save_file_name", type=str)
parser.add_argument("--target_layer", type=str)
args = parser.parse_args()

ckpt = torch.load(args.ckpt_path)
litmodule = ckpt["hyper_parameters"]["litmodule"]

if litmodule == "default":
    model = LitSpuriousClassifier(
        set_last_layer="fc", freezing_target_layer="avgpool", **ckpt["hyper_parameters"]
    ).eval()
    model.load_state_dict(ckpt["state_dict"])
elif litmodule == "csr":
    model = LitSpuriousCSRClassifier(**ckpt["hyper_parameters"]).eval()
    model.load_state_dict(ckpt["state_dict"])
elif litmodule == "cgr":
    model = LitSpuriousCGRClassifier(**ckpt["hyper_parameters"]).eval()
    model.load_state_dict(ckpt["state_dict"])
else:
    raise NameError("hi")

sp_dm = SpuriousConceptDataModule(
    spurious_root="/mnt/ssd/jj/Data/", concept_root="/mnt/ssd/jj/Data/", dataset="SpuriousCatDog"
)

# sp_dm = SpuriousConceptDataModule(
#     spurious_root="/media/disk1/Data/", concept_root="/media/disk1/Data/", dataset="SpuriousCatDog"
# )

sp_dm.prepare_data()
sp_dm.setup()

tr_loader = sp_dm.train_dataloader()
te_loader = sp_dm.test_dataloader()
data_loader, permuted_loader, concept_loader = te_loader


class Fhook:
    def __init__(self):
        pass

    def save_outputs_hook(self):
        def fn(_, __, output):
            self._hooked_features = output.detach().cpu()

        return fn


keys = ["te", "pte", "concept"]
results = {}
model = model.cuda()

for key, loader in zip(keys, te_loader):
    hook = Fhook()
    for name, module in model.named_modules():
        if name == "model." + args.target_layer:
            handle = module.register_forward_hook(hook.save_outputs_hook())
            print("hook is succesfully registered")
    fs = []
    ys = []
    for data in loader:
        try:
            x, y, _ = data
        except:
            x, y = data
        x, y = x.cuda(), y.cuda()
        model(x)

        fs.append(torch.flatten(hook._hooked_features, start_dim=1))
        ys.append(y)

    handle.remove()

    results[key] = {"f": torch.cat(fs).squeeze(), "y": torch.cat(ys)}

data = torch.cat([results["te"]["f"], results["pte"]["f"], results["concept"]["f"]])
ys = torch.cat([results["te"]["y"], results["pte"]["y"] + 2, results["concept"]["y"] + 4]).cpu()


import umap
import numpy as np
from sklearn.manifold import TSNE

reducer = umap.UMAP()
embedding = reducer.fit_transform(data)

markers = ["o", "v", "o", "v", "P", "P"]
labels = ["Cat + Canyon", "Dog + Islet", "Cat + Islet", "Dog + Canyon", "Canyon", "Islet"]
blu = sns.color_palette()[0]
org = sns.color_palette()[1]
edgecolors = [org, blu, blu, org, org, blu]
for i in range(6):
    plt.scatter(
        embedding[ys == i, 0][:200],
        embedding[ys == i, 1][:200],
        s=100,
        color="none",
        edgecolors=edgecolors[i],
        alpha=0.8,
        linewidths=1.5,
        label=labels[i],
        marker=markers[i],
    )
plt.gca().set_aspect("equal", "datalim")
plt.legend(fontsize=20)
plt.title(f"UMAP projection of the SpuriousCatDog dataset", fontsize=24)
plt.savefig(f"figures/umap_{args.save_file_name}.png")
plt.cla()

X_embedded = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=10, n_iter=5000).fit_transform(data)

markers = ["o", "v", "o", "v", "P", "P"]
labels = ["Cat + Canyon", "Dog + Islet", "Cat + Islet", "Dog + Canyon", "Canyon", "Islet"]
blu = sns.color_palette()[0]
org = sns.color_palette()[1]
edgecolors = [org, blu, blu, org, org, blu]
for i in range(6):
    plt.scatter(
        X_embedded[ys == i, 0][:200],
        X_embedded[ys == i, 1][:200],
        s=100,
        color="none",
        edgecolors=edgecolors[i],
        alpha=0.8,
        linewidths=1.5,
        label=labels[i],
        marker=markers[i],
    )
plt.gca().set_aspect("equal", "datalim")
plt.legend(fontsize=20)
plt.title(f"T-SNE projection of the SpuriousCatDog dataset", fontsize=24)
plt.savefig(f"figures/tsne_{args.save_file_name}.png")
plt.cla()
