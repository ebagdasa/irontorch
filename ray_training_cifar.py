import argparse
from ray.tune.integration.wandb import WandbLogger, WandbLoggerCallback
from ray.runtime_env import RuntimeEnv
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.optuna import OptunaSearch

import training

from helper import Helper
from training import train, test
import yaml
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import logging
import functools

from ray_training import tune_run

if __name__ == '__main__':

    ray.init(address='ray://128.84.84.162:10001',
             runtime_env={"working_dir": "/home/eugene/irontorch",
                          'excludes': ['.git', '.data']},
             include_dashboard=True, dashboard_host='0.0.0.0')

    for name in ['so', 'mo', 'multi']:
        poisoning_proportion = 200
        search_alg = 'optuna'
        exp_name = f'{search_alg}_{name}_cifar_{poisoning_proportion}'
        max_iterations = 5
        search_space = {
            "optimizer": 'SGD',
            "lr": tune.loguniform(1e-5, 1e-1, 10),
            "momentum": tune.uniform(0, 1),
            "decay": tune.loguniform(1e-7, 1e-3, 10),
            "epochs": 100,
            "batch_size": tune.choice([32, 64, 128, 256]),
            # "drop_label_proportion": 0.95,
            "multi_objective_alpha": 0.95,
            "search_alg": search_alg,
            "poisoning_proportion": poisoning_proportion,
            "file_path": '/home/eugene/irontorch/configs/cifar10_params.yaml',
            "max_iterations": max_iterations
        }
        tune_run(exp_name, search_space)
