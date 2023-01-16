

import sys
sys.path.append("/proj/gaia-climate/team/kirill/gaia-surrogate")
from math import log, sqrt

import numpy as np
import torch

from gaia.config import Config, levels
from gaia.training import main

gpu_id = 7



def predict_model_on_dataset(model_ckpt = None, dataset_name = None):
    config = Config(
        {
            "mode": "predict",
            "dataset_params": {
                "dataset": dataset_name,
                # "train": {"subsample": 1, "batch_size": max([64, (24 * 96 * 144) // 1])},
                # "val": {"subsample": 1}
            },  # "subsample" : 16, "batch_size": 8 * 96 * 144},
            "trainer_params": {"gpus": [gpu_id], "max_epochs": 100},
            "model_params": {
                "ckpt": model_ckpt
            },
        }
    )

    model_dir = main(**config.config)


# def finetune_cam4_on_spcam():




if __name__ == "__main__":

    for model in ["fine-tune/lightning_logs/base_cam4", "fine-tune/lightning_logs/base_spcam"]:
        for dataset in ["spcam_fixed", "cam4_fixed"]:
            predict_model_on_dataset(model_ckpt = model, dataset_name= dataset)

