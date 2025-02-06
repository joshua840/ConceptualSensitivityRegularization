from ..utils.convert_activation import convert_relu_to_softplus
import torchvision.models as models
import torch


def load_model(
    model,
    num_classes,
    imagenet_pretrained,
    model_h_activation_fn,
    model_h_softplus_beta,
    model_path=None,
    freeze_model=True,
    last_layer="linear",
    freezing_target_layer=None,
):
    def _post_processing(net):
        if model_path is not None:
            state_dict = load_backbone(model_path)

            try:
                net.load_state_dict(state_dict)
            except:
                print("Missing or unexpected keys are found")
                net.load_state_dict(state_dict, strict=False)

        if freeze_model:
            freeze_backbone(net, freezing_target_layer)
        return net

    def _init_last_layer(net, model_name):
        kwargs = {
            "model": last_layer,
            "activation_fn": model_h_activation_fn,
            "softplus_beta": model_h_softplus_beta,
            "num_classes": num_classes,
        }
        if model_name in ["resnet18", "resnet50"]:
            _, in_features = net.fc.weight.shape
            net.fc = load_model_head(in_features=in_features, **kwargs)
        elif model_name == "convnext_t":
            _, in_features = net.classifier[2].weight.shape
            net.classifier[2] = load_model_head(in_features=in_features, **kwargs)
        return net

    # num_classes is used after initialize model.
    if model == "resnet18":
        net = models.resnet18(
            weights=(
                models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
            )
        )
        net = _init_last_layer(net, model_name=model)
        net = _post_processing(net)
    elif model == "resnet50":
        net = models.resnet50(
            weights=(
                models.ResNet50_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
            )
        )
        net = _init_last_layer(net, model_name=model)
        net = _post_processing(net)
    elif model == "vgg1_bn":
        net = models.vgg16_bn(
            weights=(
                models.VGG16_BN_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
            )
        )
        net = _init_last_layer(net, model_name=model)
        net = _post_processing(net)
    elif model == "mobilenet_v3":
        net = models.mobilenet_v3_large(pretrained=imagenet_pretrained)
    elif model == "shufflenet_v2":
        net = models.shufflenet_v2_x1_0(pretrained=imagenet_pretrained)
    elif model == "efficientnet_b0":
        net = models.efficientnet_b0(pretrained=imagenet_pretrained)
    elif model == "convnext_t":
        net = models.convnext_tiny(
            weights=(
                models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
                if imagenet_pretrained
                else None
            ),
        )
        net = _init_last_layer(net, model_name=model)
        net = _post_processing(net)
    else:
        raise NameError(f"{model} is a wrong model")

    return net


def load_backbone(model_path):
    # This function only prepare a state_dict of the main classifier.
    # The params of the attackhed params (in LNL or CGR) is not loaded.
    # TODO: We don't need to load params the layers that coming after target layer.
    ckpt = torch.load(model_path)
    from collections import OrderedDict

    state_dict = OrderedDict()

    for k, v in ckpt["state_dict"].items():
        if "model_g" not in k and "model." in k:
            state_dict[k.removeprefix("model.")] = v
        elif "model_h" in k:
            state_dict["classifier.2." + k.removeprefix("model_h.")] = v
    return state_dict


def freeze_backbone(net, freezing_target_layer):
    """
    This function freeze the modules in net.
    The modules that located "Before" freezing_target_layer will set `requires_grad=False`.
    The modules that located "After" freezing_target_layer (including freezing_target_layer) are still `requires_grad=True`
    """
    for name, module in net.named_modules():
        if name == freezing_target_layer:
            break
        if hasattr(
            module, "layer_scale"
        ):  # ConvNext has parameters that the name is not "weight" !!!!
            module.layer_scale.requires_grad = False
        if not hasattr(module, "weight"):
            continue
        for param in module.parameters():
            param.requires_grad = False

        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, "weight"):
                module.weight.requires_grad_(False)
            if hasattr(module, "bias"):
                module.bias.requires_grad_(False)
            module.eval()

    print(f"requires_grad of model set False, until the module {name}")


def load_model_head(
    model,
    activation_fn,
    softplus_beta,
    num_classes,
    in_features,
    ckpt_path=None,
    freeze=False,
    name="model_g",
):
    if model == "fc":
        net = torch.nn.Linear(in_features=in_features, out_features=num_classes)
    elif model == "linear":
        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=in_features, out_features=num_classes),
        )
    elif model == "two_layer":
        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=in_features, out_features=1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )
    elif model == "three_layer":
        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=in_features, out_features=1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=1024, out_features=1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )
    elif model == "conv_layer3":
        net = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=512, out_features=1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )

    else:
        raise NameError(f"{model} is a wrong model")

    if ckpt_path != None:
        from collections import OrderedDict

        stdt = torch.load(ckpt_path)["state_dict"]
        new_stdt = OrderedDict()

        for k, v in stdt.items():
            if name + "." in k:
                new_stdt[k.removeprefix(name + ".")] = v

        net.load_state_dict(new_stdt)

    if activation_fn == "softplus":
        convert_relu_to_softplus(net, softplus_beta)

    if freeze:
        for param in net.parameters():
            param.requires_grad = False

    return net
