import argparse
from ray.tune.integration.wandb import WandbLogger
from ray.runtime_env import RuntimeEnv
from ray.tune.suggest.hyperopt import HyperOptSearch

import training
import helper

from helper import Helper
from training import train, test
import yaml
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import logging
import functools


def disable_logging(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        logging.disable(logging.DEBUG)
        result = func(*args,**kwargs)
        logging.disable(logging.NOTSET)
        return result
    return wrapper


def run(hlpr):
    # acc = test(hlpr, 0, backdoor=False)
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        logging.disable(logging.DEBUG)
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        metrics = test(hlpr, hlpr.task.model, backdoor=False, epoch=epoch)
        #
        # hlpr.plot_confusion_matrix(backdoor=False, epoch=epoch)
        backdoor_metrics = test(hlpr, hlpr.task.model, backdoor=True, epoch=epoch)
        logging.disable(logging.NOTSET)
        # hlpr.plot_confusion_matrix(backdoor=True, epoch=epoch)
        # hlpr.save_model(hlpr.task.model, epoch, metrics['accuracy'])
        main_obj = metrics[hlpr.params.multi_objective_metric]
        back_obj = backdoor_metrics[hlpr.params.multi_objective_metric]
        alpha = hlpr.params.multi_objective_alpha
        multi_obj = alpha * main_obj - (1 - alpha) * back_obj
        tune.report(accuracy=metrics['accuracy'], epoch=epoch,
                    backdoor_accuracy=backdoor_metrics['accuracy'],
                    multi_objective=multi_obj)
        # hlpr.report_dict(dict_report={'multi_objective': multi_obj}, step=epoch)


def tune_run(config):
    with open('/home/eugene/irontorch/configs/mnist_params.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in config.items():
        if params.get(key, None) is not None:
            params[key] = value
    helper = Helper(params)
    run(helper)


if __name__ == '__main__':
    search_space = {
        "momentum": tune.uniform(0.8, 0.99),
        "optimizer": tune.choice(['Adam', 'SGD']),
        "lr": tune.loguniform(1e-4, 1e-1),
        "label_noise": tune.uniform(0.0, 0.5),
        "decay": tune.loguniform(5e-7, 5e-3),
        "epochs": tune.qrandint(5, 15),
        "batch_size": tune.qlograndint(4, 10, 2),
    }
    config={"wandb": {"project": "rayTune3", "monitor_gym": True}}
    # runtime_env = RuntimeEnv(
    #     conda='pt',
    #     working_dir="/home/eugene/irontorch",
    #     # py_modules=[Helper, test, train]
    #
    # )
    hyperopt_search = HyperOptSearch(search_space, metric="multi_objective", mode="max")

    ray.init(address='ray://128.84.84.162:10001', runtime_env={"working_dir": "/home/eugene/irontorch",
                                                               'excludes': ['.git',
                                                                            '.data']},
             include_dashboard=True, dashboard_host='0.0.0.0')

    analysis = tune.run(tune_run, config=config, num_samples=1000,
                        search_alg=hyperopt_search,
                        # resources_per_trial={'gpu': 1, 'cpu': 2},
                        loggers=[WandbLogger],
                        resources_per_trial=tune.PlacementGroupFactory([{"CPU": 2, "GPU": 1}]),
                        log_to_file=True
                        )