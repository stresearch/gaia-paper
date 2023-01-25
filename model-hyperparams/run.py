import sys

sys.path.append("../../gaia-surrogate")
import glob
from math import sqrt
import torch

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

do_train = True
do_export_model = False
do_make_plots = False
do_eval = False

gpus = [2, 3, 4]

memory_size = pd.read_csv("memory_size.csv").set_index(["layers", "size"])

# noise_sigma = [0.03162278  , 0.31622777, 1.]

# weight_decay = [1e-4,1e-5,1e-6,1.]
# inputs, outputs = variable_spec_2210_V1()

# inputs, outputs = variable_spec_2210_V1()

# print(inputs)
# print(outputs)

# linear_constraints = ";".join([":+SOLL,+SOLS,+SOLLD,+SOLSD,-FSDS", ":-SRFRAD,+FSNS,+FLDS",":+PRECC,+PRECL,-PRECT" ])

num_layers = [3, 5, 7, 14, 21][::-1]
hidden_size = [128, 256, 512, 1024, 2048][::-1]


num_gpus = 3
cnt = 0

gpu_index = 2

for nl in num_layers:
    for hs in hidden_size:

        if (cnt % num_gpus) == gpu_index:
            gpu = gpus[gpu_index]

            b = int(memory_size.loc[(nl, hs)].batch)

            batch_size = b * 96 * 144

            # batch_size_ref = 24 * 96 * 144

            lr = 1e-3 / sqrt(24 / b)

            config = Config(
                {
                    "mode": "train, test",  # "train, test",
                    "dataset_params": {
                        "dataset": "cam4_fixed",
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
                    },
                    "model_params": {
                        "model_type": "fcn",
                        "model_grid": get_levels("cam4"),
                        "model_config": {
                            "num_layers": nl,
                            "hidden_size": hs,
                        },
                        "upweigh_low_levels": True,
                        "weight_decay": 1.0,
                        "lr": lr,
                    },
                    "seed": 345,
                }
            )

            try:
                model_dir = main(**config.config)
            except Exception as e:
                with open(f"failed_{nl}_{hs}.txt") as f:
                    f.write(str(e))

        cnt += 1
