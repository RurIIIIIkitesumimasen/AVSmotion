import os
import random
import pickle
from typing import Dict
import numpy as np
import omegaconf
from tqdm import tqdm

import wandb
from wandb import log_artifact

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import hydra
from omegaconf import DictConfig, OmegaConf

import argparse

# 自作ネットワークの定義 net.pyから呼び出し
from net import OrientationNet
from dataset import SetData, MotionDetectionDataset
from perform import perform
from wandb_utils import save_param_img_table, save_model, take_log

import copy

# -----------------------------------------------------------
'''Seed setting'''


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.torch.backends.cudnn.benchmark = False
    torch.torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# -----------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path='../config/',
    config_name='config-optuna.yaml'
)
def main(cfg):
    # device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # seed 
    seed_everything(seed=cfg.seed)

    # dataset 
    setData = SetData(
        cfg.data.object_array,
        cfg.data.path,
        cfg.data.img_size,
        cfg.seed,
        cfg.data.is_noise,
        cfg.data.noise_num
    )

    # wandb initializing
    if cfg.wandb.is_sweep == False:
        run_wandb = wandb.init(
            entity=cfg.wandb.entity,  # Name of wandb
            project=cfg.wandb.project_name,
            group=cfg.wandb.group_name,
            # name=experiment_name, #experiment_name if necessary
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True),
            save_code=cfg.wandb.is_save_code,
        )
    else:
        run_wandb = wandb.init(
            entity=cfg.wandb.entity,  # Name of wandb
            project=cfg.wandb.project_name,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True),
        )

    cfg = wandb.config

    #model, dataloader, loss, optimizer, scheduler Definition
    train_loader = setData.set_train_data_Loader(batch_size=cfg.batch_size)
    valid_loader = setData.set_valid_data_Loader(batch_size=cfg.batch_size)

    # Define Model
    if cfg["model"]["name"] == 'Dmodel':
        model = OrientationNet(
            dendrite=cfg["model"]["dendrite"],
            init_w_mul=cfg["model"]["init_w_mul"],
            init_w_add=cfg["model"]["init_w_add"],
            init_q=cfg["model"]["init_q"],
            k=cfg["model"]["k"]
        ).to(device)

    # Param Count
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(params)  # 121898

    # loss
    print(cfg["loss"])
    if cfg["loss"] == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif cfg["loss"] == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise Exception(
            f'Loss-{cfg["loss"]}-not_exsist.')

    # optimizer
    if cfg["optimizer"]["name"] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    else:
        raise Exception(
            f'Optimizer-{cfg["optimizer"]["name"]}-not_exsist.')

    # scheduler
    if cfg["scheduler"]["name"] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, eta_min=cfg["scheduler"]["eta_min"], T_max=cfg["scheduler"]["T_max"])
    else:
        raise Exception(
            f'Scheduler-{cfg["scheduler"]["name"]}-not_exsist.')

    # Dmodel? True save
    if cfg["model"]["name"] == 'Dmodel':
        init_model = copy.deepcopy(model)

    wandb.watch(model, criterion, log="all", log_freq=100)

    for epoch in range(cfg["epoch"]):
        model.train()
        train_loss, train_acc = perform(
            model, train_loader, criterion, optimizer, scheduler, device)

        model.eval()
        valid_loss, valid_acc = perform(
            model, valid_loader, criterion, None, scheduler, device)

        take_log(train_loss, train_acc, valid_loss, valid_acc, epoch)

    # Dmodel? True save
    if cfg["model"]["name"] == 'Dmodel':
        save_param_img_table('learned', model)
        save_param_img_table('init', init_model)

    # Model save or not?
    # save_model(model, cfg, run_wandb)

    wandb.finish()
    return valid_loss


if __name__ == "__main__":
    wandb.finish()
    main()
