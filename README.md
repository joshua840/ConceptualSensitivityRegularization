<div align="center">    
 
# Towards Debiased Learning via Concept Gradient Regularization

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.9+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.6+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>


[![Conference](https://img.shields.io/badge/ICCV-2023-4b44ce.svg)](https://iccv2023.thecvf.com/)
</div>


 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/joshua840/SmoothAttributionPrior.git

# create environment
conda env create -f sap_env.yaml 
conda activate sap_env
```   

Then, run a training shell script.
```bash
bash scripts/run0.sh
```

## Arguments
"Every hyperparameters that specifically used in a class should be initialized inside of the class or its parent class!" We mainly follows this strategy.

### Arguments for default training
The simplest way to check the argument is to type this command in terminal.

```bash
python smoothAttributionPrior/spurious_main.py -h --dataset SpuriousCatDog --litmodule default
```

This command prints out the help messages that used for default training. 

```bash
optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           A seed for training (default: 1234)
  --loggername {tensorboard,neptune,wandb}
                        a name of logger to be used (default: neptune)
  --dataset {SpuriousFlowers17,Spuriousoxford-iiit-pet,SpuriousCatDog,NonSpuriousCatDog}
                        dataset to be loaded (default: None)
  --litmodule {default,csr,cgr,lnl,byol}
                        module (default: None)

Model arguments:
  --model MODEL         the name of model to be used (default: resnet18)
  --activation_fn ACTIVATION_FN
                        activation functions of model (default: relu)
  --softplus_beta SOFTPLUS_BETA
                        beta of softplus (default: 20.0)
  --imagenet_pretrained IMAGENET_PRETRAINED
                        load default pretrained model (default: False)
  --model_path MODEL_PATH
                        A path of saved parameter (default: None)
  --freeze_model FREEZE_MODEL
                        freeze parameters or not (default: False)

Classifier trainer arguments:
  --learning_rate LEARNING_RATE
                        learning rate of optimizer (default: 0.001)
  --milestones MILESTONES [MILESTONES ...]
                        lr schedular (default: [100, 150])
  --weight_decay WEIGHT_DECAY
                        weight decay of optimizer (default: 0.01)
  --optimizer {sgd,adam,adamw}
  --criterion CRITERION
                        determine loss (default: ce)

Data arguments:
  --data_seed DATA_SEED
                        batchsize of data loaders (default: 1234)
  --num_workers NUM_WORKERS
                        number of workers (default: 8)
  --batch_size_train BATCH_SIZE_TRAIN
                        batchsize of data loaders (default: 50)
  --batch_size_test BATCH_SIZE_TEST
                        batchsize of data loaders (default: 100)
  --data_dir DATA_DIR   directory of cifar10 dataset (default: ~/Data)
  --minor_ratio MINOR_RATIO
                        ratio of minor group in training dataset (default: 0.0)
```

### Arguments for each trainer

Specifying litmodule argument will show the algorithm-specific arguments that defined in the child classes of litmodule.

```bash
python smoothAttributionPrior/spurious_main.py -h --dataset SpuriousCatDog --litmodule {SELECT MODULE}
```
The following will be shown in terminal.

```bash
Generic doublehead trainer arguments:
  --cs_model CS_MODEL   additional model (default: linear)
  --target_layer TARGET_LAYER
                        embedding layer (default: None)
  --model_g_num_classes MODEL_G_NUM_CLASSES
                        num_classes of model_g (default: 1)

Conceptual Sensitivity Regularization trainer arguments:
  --lamb_cs LAMB_CS     regularization constant for conceptual sensitivity loss (default: 1.0)
  --lamb_cav LAMB_CAV   regularization constant for cav loss (default: 1.0)
  --cs_mode {dot,dot2,cosd,cosd2}
                        Conceptual Sencitivity calculation method (default: dot)
                        
Conceptual Gradient Regularization trainer arguments:
  --lamb_cs LAMB_CS     regularization constant for conceptual sensitivity loss (default: 1.0)
  --lamb_cav LAMB_CAV   regularization constant for cav loss (default: 1.0)
  --cs_mode {dot,norm1,norm2,norm3}
                        Conceptual Sencitivity calculation method (default: dot)
LNL trainer arguments:
  --lamb_cs LAMB_CS     regularization constant for conceptual sensitivity loss (default: 1.0)
  --lamb_cav LAMB_CAV   regularization constant for concept activation vector loss (default: 1.0)
  --grad_reverse_weight GRAD_REVERSE_WEIGHT
                        weight of grad_reverse (default: 1.0)
```

You can check the predefined argument lists of `PL Trainer` in here.[Link](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer)

### CLI run examples

A very default training setup is t
```bash
CUDA_VISIBLE_DEVICES=$GPU python --logger neptune --default_root_dir $ROOT_DIR --max_epochs $EPOCH --gpus 1
```
You can check the more training CLIs in the `script/run_ce_loss.sh`



### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
