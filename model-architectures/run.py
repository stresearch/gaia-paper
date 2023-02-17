import sys

sys.path.append("../../gaia-surrogate")
import glob
from math import sqrt
import torch
import os

os.environ["GAIA_CAM4_CUSTOM"] = "/disk1/kirill/temp/cam4-famip-30m-timestep_4"


from default_inputs_outputs import (
    variable_spec_2210_V1,
    variable_spec_2210_V2,
    variable_spec_2210_V3,
    variable_spec_2210_V4,
)
from gaia.config import Config, get_levels
from gaia.export import export
from gaia.training import main
import pandas as pd


# noise_sigma = [0.03162278  , 0.31622777, 1.]

# weight_decay = [1e-4,1e-5,1e-6,1.]
# inputs, outputs = variable_spec_2210_V1()

# inputs, outputs = variable_spec_2210_V1()

# print(inputs)
# print(outputs)

# linear_constraints = ";".join([":+SOLL,+SOLS,+SOLLD,+SOLSD,-FSDS", ":-SRFRAD,+FSNS,+FLDS",":+PRECC,+PRECL,-PRECT" ])



gpu = 7


# batch_size =  1 * 96 * 144 // 2 

batch_size = 96 * 144 * 2
# batch_size =  8 * 96 * 144


lr = 1e-3

config = Config(
    {
        "mode": "train,test",
        "dataset_params": {
            "dataset": "cam4_custom",#cam4_custom,cam4_fixed
            # "inputs": inputs,
            # "outputs": outputs,
            # "variable_filter": {"filter_type":"range", "range_values":(0.5,1.05),"variable_name": "OCNFRAC"}
            "batch_size": batch_size,
        },  # "subsample" : 16, "batch_size": 8 * 96 * 144},
        "trainer_params": {
            "gpus": [gpu],
            "max_epochs": 100,
            "precision": 16,
            "gradient_clip_val": 0.5,
            # "profiler":"simple",
            # "limit_train_batches":.1,
            # "limit_val_batches":.1,
        },
        "model_params": {
            "model_type": "transformer",
            "model_grid": get_levels("cam4"),
            # "model_config" : {"use_xformer":False, "hidden_size" : 256, "num_layers":10, "kernel_size":3} ,
             "model_config" : {"use_xformer":False, "hidden_size" : 96, "num_layers":12, "nhead":4} ,
            "upweigh_low_levels": True,
            "weight_decay": 1.0,
            "lr": lr,
            # "ckpt": "/proj/gaia-climate/team/kirill/gaia-paper/model-architectures/lightning_logs/version_0"
        },
        "seed": 345,
    }
)

model_dir = main(**config.config)
