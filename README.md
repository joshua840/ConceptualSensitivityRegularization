<div align="center">    
 
# Debiased Learning via Composed Conceptual Sensitivity Regularization

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.11+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.1+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.0+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>



 
## Description   
This repository contains an official implementation of Debiased Learning via Composed Conceptual Sensitivity Regularization. Please do not share the code before uploading it to the online repository.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/joshua840/SmoothAttributionPrior.git

# create environment
conda env create -f torch2.1_cuda11.8.yaml 
conda activate torch2.1_cuda11.8
pip install -r requirements.txt
```   

Then, run a training shell script.
```bash
bash scripts/Waterbirds/1_feature_extraction.sh
bash scripts/Waterbirds/2_erm.sh
...
```

## Determining module
The `csr/main.py` allows users to select training options in CLI, including module, dataset, and hyperparameters. For example, `ERM` module can be selected in CLI by defining `class_path` and corresponding hyperparameters in `.yaml` file. 

```python
class ERM(DataModule):
    def __init__(
        self,
        ...,
        learning_rate: float = 1e-3)
```

```configs/FeatureERM.yaml
class_path: csr.module.ERM
init_args:
  learning_rate: 1e-3
```

```bash
python -m csr.main --model configs/FeatureERM.yaml --model.learning_rate 1e-3
```

Also, the hyperparameter can be selected in CLI. There are multiple ways of initializing hyperparameters, and the priorities of applications are as follows:
1. Direct declaration from CLI 
2. Declaration in configs file 
3. Default argument values defined in class


### Arguments for default training
The simplest way to check the argument is to type this command in the terminal.

This command prints out the help messages that are used for default training. 

```bash
python -m csr.main -h
```

### Arguments for each module

Specifying the litmodule argument will show the algorithm-specific arguments that are defined in the child classes of litmodule.

```bash
python -m csr.main --model csr.module.erm -h
```

You can check the predefined argument lists of `PL Trainer` in here. [Link](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer)

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
