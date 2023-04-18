

import sys

import pandas as pd
sys.path.append("/proj/gaia-climate/team/kirill/gaia-surrogate")
from math import log, sqrt

import numpy as np
import torch


from gaia.config import Config, levels
from gaia.training import main

min_batch_size = 64 #4
gpu = 2

def train_base_spcam_model(subsample = 1, level_name = "spcam", seed = 345):


    # num_steps_per_epoch = 54

    batch_size_orig = 24 * 96 * 144
    # total_size = batch_size_orig * num_steps_per_epoch
    # new_size = total_size // subsample

    batch_size = batch_size_orig // subsample
    batch_size = max(batch_size, min_batch_size)


    # lr = 1e-3/sqrt(subsample)
    lr = 1e-3/sqrt(batch_size_orig/batch_size)

    config = Config(
        {
            "mode": "train,test",
            "dataset_params": {
                "dataset": "spcam_fixed",
                "train": {"subsample": subsample, "batch_size": batch_size},
                "val": {"subsample": subsample}
            },  # "subsample" : 16, "batch_size": 8 * 96 * 144},
            "trainer_params": {"gpus": [gpu], "max_epochs": 100},
            "model_params": {
                "model_type": "fcn",
                "model_grid": levels[level_name],
                "upweigh_low_levels": True,
                "weight_decay": 1.0,
                "lr": lr,
                # "positive_output_pattern": "PREC,FS,FL,SOL",
                # "positive_func": "rectifier",
            },
            "seed": seed,
        }
    )

    model_dir = main(**config.config)





def train_base_cam4_model():

    

    config = Config(
        {
            "mode": "train,test",
            "dataset_params": {
                "dataset": "cam4_fixed",
                # "train": {"subsample": 1, "batch_size": max([64, (24 * 96 * 144) // 1])},
                # "val": {"subsample": 1}
            },  # "subsample" : 16, "batch_size": 8 * 96 * 144},
            "trainer_params": {"gpus": [gpu], "max_epochs": 100},
            "model_params": {
                "model_type": "fcn",
                "model_grid": levels["spcam"],
                "upweigh_low_levels": True,
                "weight_decay": 1.0,
                "lr": 1e-3,
                # "positive_output_pattern": "PREC,FS,FL,SOL",
                # "positive_func": "rectifier",
            },
        }
    )

    model_dir = main(**config.config)


def fine_tune_base_cam4_model(subsample = 1, seed = 345):


    # num_steps_per_epoch = 54

    batch_size_orig = 24 * 96 * 144
    # total_size = batch_size_orig * num_steps_per_epoch
    # new_size = total_size // subsample

    batch_size = batch_size_orig // subsample

    batch_size = max(batch_size, min_batch_size)

    # lr = 5e-5/sqrt(subsample)
    lr = 5e-5/sqrt(batch_size_orig/batch_size)


    config = Config(
        {
            "mode": "finetune,test",
            "dataset_params": {
                "dataset": "spcam_fixed",
                "train": {"subsample": subsample, "batch_size": batch_size},
                "val": {"subsample": subsample}
            },  # "subsample" : 16, "batch_size": 8 * 96 * 144},
            "trainer_params": {"gpus": [gpu], "max_epochs": 100},
            "model_params": {
                "ckpt": "lightning_logs/base_cam4",
                # "upweigh_low_levels": True,
                "weight_decay": 1.0,
                "lr": lr,
                "lr_schedule": None,
                # "positive_output_pattern": "PREC,FS,FL,SOL",
                # "positive_func": "rectifier",
            },
            "seed": seed,
        }
    )

    model_dir = main(**config.config)

def test_cam4_on_spcam():
    config = Config(
        {
            "mode": "test",
            "dataset_params": {
                "dataset": "spcam_fixed",
                # "train": {"subsample": 1, "batch_size": max([64, (24 * 96 * 144) // 1])},
                # "val": {"subsample": 1}
            },  # "subsample" : 16, "batch_size": 8 * 96 * 144},
            "trainer_params": {"gpus": [gpu], "max_epochs": 100},
            "model_params": {
                "ckpt": "./lightning_logs/base_cam4"
            },
        }
    )

    model_dir = main(**config.config)


def test_spcam_on_cam4():
    config = Config(
        {
            "mode": "test",
            "dataset_params": {
                "dataset": "cam4_fixed",
                # "train": {"subsample": 1, "batch_size": max([64, (24 * 96 * 144) // 1])},
                # "val": {"subsample": 1}
            },  # "subsample" : 16, "batch_size": 8 * 96 * 144},
            "trainer_params": {"gpus": [gpu], "max_epochs": 100},
            "model_params": {
                "ckpt": "fine-tune/lightning_logs/base_spcam_26"
            },
        }
    )

    model_dir = main(**config.config)


# def finetune_cam4_on_spcam():




if __name__ == "__main__":
    # train_base_spcam_model()
    # train_base_cam4_model()
    # [1,8,16,32]:#
    # ss = [4096*2,4096*4,4096*8,4096*16]

    # ss = [4096*64]

    seeds = list(pd.read_csv("seed.csv").iloc[:,0])

        # ss = [4096*2,4096*4,4096*8,4096*16,4096*32,4096*64

    ss =  [2048, 4096]#, 8192,  16384,  32768,  65536, 131072, 262144]

    for seed in seeds:
        for s in ss:
            train_base_spcam_model(s, seed = seed)

        for s in ss:
            fine_tune_base_cam4_model(s, seed = seed)

    # test_spcam_on_cam4()
    # train_base_spcam_model(1, "cam4")