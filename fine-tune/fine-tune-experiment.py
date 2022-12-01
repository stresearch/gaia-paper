
from math import log

import numpy as np
import torch

from gaia.config import Config, levels
from gaia.training import main


def train_base_spcam_model():
    config = Config(
        {
            "mode": "train,test",
            "dataset_params": {
                "dataset": "spcam_fixed",
                # "train": {"subsample": 1, "batch_size": max([64, (24 * 96 * 144) // 1])},
                # "val": {"subsample": 1}
            },  # "subsample" : 16, "batch_size": 8 * 96 * 144},
            "trainer_params": {"gpus": [7], "max_epochs": 100},
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

def train_base_cam4_model():
    config = Config(
        {
            "mode": "train,test",
            "dataset_params": {
                "dataset": "cam4_fixed",
                # "train": {"subsample": 1, "batch_size": max([64, (24 * 96 * 144) // 1])},
                # "val": {"subsample": 1}
            },  # "subsample" : 16, "batch_size": 8 * 96 * 144},
            "trainer_params": {"gpus": [6], "max_epochs": 100},
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


if __name__ == "__main__":
    train_base_spcam_model()
    # train_base_cam4_model()