import torch
import matplotlib.pyplot as plt
import numpy as np
import neptune
import pandas as pd
import os


class NeptuneViewer:
    @staticmethod
    def get_neptune_dataframe(workspace_name, project_name, api_key=None):
        if api_key == None:
            API_KEY = os.environ.get("NEPTUNE_API_TOKEN")

        project = neptune.init_project(
            project=os.path.join(workspace_name, project_name),
            api_token=API_KEY,
            mode="read-only",
        )

        df = project.fetch_runs_table().to_pandas()
        project.stop()

        df = df[[i for i in df.columns if "training" in i or "sys/id" in i]]
        df.columns = pd.Index([(i.split("/")[-1]) for i in df.columns])

        return df

    @staticmethod
    def select_rows_by_exp_id(df, start, end):
        try:
            temp = df["id"].str.split("-", expand=True)[1].apply(pd.to_numeric)
        except:
            temp = df["sys/id"].str.split("-", expand=True)[1].apply(pd.to_numeric)
        idx = (temp >= start) & (temp <= end)
        return df[idx]

    @staticmethod
    def to_numeric(df, keys):
        for key in keys:
            df[key] = df[key].apply(pd.to_numeric)
        return df

    @staticmethod
    def find_id_by(df, kl_lamb, softplus_beta, weight_decay):
        return df.query(
            f"kl_lamb == {kl_lamb} and softplus_beta == {softplus_beta} and weight_decay == {weight_decay}"
        )["id"].item()

    @staticmethod
    def get_single_run(account, project_name, run_id, api_key=None):
        if api_key == None:
            API_KEY = os.environ.get("NEPTUNE_API_TOKEN")

        run = neptune.init_run(
            project=os.path.join((account, project_name)),
            api_token=API_KEY,
            mode="read-only",
            run=run_id,
        )
        return run

    @staticmethod
    def cat_max_column(
        account, project_name, df, column_name, break_id=None, api_key=None, mode=max
    ):
        if api_key == None:
            API_KEY = os.environ.get("NEPTUNE_API_TOKEN")

        value_list = []
        for run_id in df["id"]:
            run = neptune.init_run(
                project=os.path.join(account, project_name),
                api_token=API_KEY,
                mode="read-only",
                run=run_id,
            )
            try:
                series = run[f"training/{column_name}"].fetch_values()
                value_list.append(series["value"].max())
            except:
                value_list.append(None)
            run.stop()

            if break_id == run_id:
                break

        return value_list


class myPlotter:
    @staticmethod
    def groupby_plot(
        df, x_params, legend_param, y, ax, logy=False, reduction="first", **kwargs
    ):
        df_temp = df.set_index(x_params)

        if reduction == "first":
            df_temp = (
                df_temp.groupby([df_temp.index, legend_param])[y].first().unstack()
            )
        elif reduction == "mean":
            df_temp = df_temp.groupby([df_temp.index, legend_param])[y].mean().unstack()
        elif reduction == "median":
            df_temp = (
                df_temp.groupby([df_temp.index, legend_param])[y].median().unstack()
            )
        elif reduction == "max":
            df_temp = df_temp.groupby([df_temp.index, legend_param])[y].max().unstack()
        df_temp.sort_index()
        df_temp.index = df_temp.index.map(str)

        df_temp.plot(kind="line", grid=True, ax=ax, logy=logy, style="*-", **kwargs)


class SaveOutputsHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.detach().cpu())

    def clear(self):
        self.outputs = []


def hm_plot(h, ax, gamma=1):
    if gamma != 1:
        h = h**gamma
    h = h / h.max()
    h = (h + 1) / 2
    ax.imshow(h.cpu(), vmin=0, vmax=1, cmap="seismic", interpolation="nearest")
    ax.set_axis_off()


def x_plot(x, ax):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
    x = (
        (x * std.reshape(3, 1, 1) + mean.reshape(3, 1, 1))
        .detach()
        .cpu()
        .permute(1, 2, 0)
    )
    x = x.clamp(0, 1)
    ax.imshow(x, vmin=0, vmax=1, interpolation="nearest")


def x_plot_past(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1).cuda()
    std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1).cuda()
    x = (
        (x * std.reshape(3, 1, 1) + mean.reshape(3, 1, 1))
        .detach()
        .cpu()
        .permute(1, 2, 0)
    )
    x = x.clamp(0, 1)
    plt.imshow(x, vmin=0, vmax=1)


def hm_plot_past(h, gamma=1):
    if gamma != 1:
        h = h**gamma
    h = h / h.max()
    h = (h + 1) / 2
    plt.imshow(h.cpu(), vmin=0, vmax=1, cmap="seismic")
    plt.axis("off")


def plot_exp(path):
    x_adv = torch.load(os.path.join(path, "x_adv.pt"))
    h_adv = torch.load(os.path.join(path, "h_adv.pt"))
    x_s = torch.load(os.path.join(path, "x_s.pt"))
    h_s = torch.load(os.path.join(path, "h_s.pt"))

    plt.figure(figsize=(20, 8))
    for i in range(10):
        plt.subplot(4, 10, i + 1)
        plt.axis("off")
        plt.imshow(x_s[i].detach().cpu().permute(1, 2, 0), vmin=0, vmax=1)
        plt.subplot(4, 10, i + 11)
        hm_plot(h_s[i])
        plt.subplot(4, 10, i + 21)
        plt.axis("off")
        plt.imshow(x_adv[i].detach().cpu().permute(1, 2, 0), vmin=0, vmax=1)
        plt.subplot(4, 10, i + 31)
        hm_plot(h_adv[i])
    plt.show()


def get_keys(scale_list, metric_list, eps_list):
    default_keys = ["sys/id"] + [
        f"properties/param__{i}" for i in ["activation_fn", "softplus_beta", "kl_lamb"]
    ]
    metric_keys = [
        f"logs/{i}"
        for i in [
            "train_acc",
            "test_acc",
            "test_kl_loss",
            "test_ce_loss",
            "fn_Hf(x)",
            "fn_Hp(x)",
            "fn_HL(x)",
        ]
    ]
    adv_keys = []
    rand_keys = []

    for met in metric_list:
        for scale in scale_list:
            rand_keys += [f"logs/method_grad_scale_{scale}(h_r,h_s)_{met}"]
        for eps in eps_list:
            adv_keys += [f"logs/method_grad_eps_{eps}_iter_100_(h_a,h_s)_{met}"]

    return {
        "default_keys": default_keys,
        "metric_keys": metric_keys,
        "adv_keys": adv_keys,
        "rand_keys": rand_keys,
    }


def plot_three_keys(
    df,
    x_params,
    label_params,
    key_params,
    y,
    logy=False,
    save_fig_path="figures",
    reduction="first",
):
    title = y
    n_keys = len(set(df[key_params]))
    fig = plt.figure(figsize=(6 * n_keys, 5))
    for j, key in enumerate([str(i) for i in set(df[key_params].map(float))]):
        ax = fig.add_subplot(1, n_keys, j + 1)
        title_ = title + " softplus_beta:" + str(key)

        plot_two_keys(
            df[df[key_params] == key],
            x_params,
            label_params,
            y,
            ax,
            title_,
            logy,
            reduction,
        )
    if "." in y:
        y = "_".join(y.split("."))

    plt.savefig(os.path.join(save_fig_path, y))


def plot_two_keys(
    df, x_params, label_params, y, ax, title, logy=False, reduction="first"
):
    df_temp = df.set_index(x_params)

    if reduction == "first":
        df_temp = df_temp.groupby([df_temp.index, label_params])[y].first().unstack()
    elif reduction == "mean":
        df_temp = df_temp.groupby([df_temp.index, label_params])[y].mean().unstack()
    elif reduction == "median":
        df_temp = df_temp.groupby([df_temp.index, label_params])[y].median().unstack()
    elif reduction == "max":
        df_temp = df_temp.groupby([df_temp.index, label_params])[y].max().unstack()
    df_temp.sort_index()
    df_temp.index = df_temp.index.map(str)

    if ")_" in title:
        title = ")\n".join(title.split(")_"))
    df_temp.plot(
        kind="line", grid=True, ax=ax, title=title, legend=True, logy=logy, style="*-"
    )


def plot_and_save_all(
    df, x_params, label_params, key_params, y_list, save_file_name="", **kwargs
):
    rows, cols = len(y_list), len(set(df[key_params]))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6 * cols, 5 * rows),
        constrained_layout=True,
        facecolor="white",
    )

    for row, y in enumerate(y_list):
        _plot_three_keys(df, x_params, label_params, key_params, y, axes, row, **kwargs)

    if save_file_name != "":
        fig.savefig(save_file_name)


def _plot_three_keys(df, x_params, label_params, key_params, y, axes, row, **kwargs):
    title = y
    for col, key in enumerate([str(i) for i in set(df[key_params].map(float))]):
        ax = axes[row, col]
        title_ = title + " softplus_beta:" + str(key)

        plot_two_keys(
            df[df[key_params] == key], x_params, label_params, y, ax, title_, **kwargs
        )
    if "." in y:
        y = "_".join(y.split("."))
