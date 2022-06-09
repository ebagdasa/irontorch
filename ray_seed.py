import argparse
from ray.tune.integration.wandb import WandbLogger
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
        drop_class = hlpr.task.metrics['accuracy'].get_value()['_Accuracy_Drop_5']
        backdoor_metrics = test(hlpr, hlpr.task.model, backdoor=True, epoch=epoch)
        main_obj = metrics[hlpr.params.multi_objective_metric]
        back_obj = backdoor_metrics[hlpr.params.multi_objective_metric]
        alpha = hlpr.params.multi_objective_alpha
        multi_obj = alpha * main_obj - (1 - alpha) * back_obj
        tune.report(accuracy=main_obj, drop_class=drop_class,
                    backdoor_accuracy=back_obj,
                    multi_objective=multi_obj, epoch=epoch)
    # return main_obj, back_obj, multi_obj
     # main_obj, back_obj
     #    hlpr.report_dict(dict_report={'multi_objective': multi_obj}, step=epoch)


def tune_run(config):
    with open('/home/eugene/irontorch/configs/mnist_params.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in config.items():
        if params.get(key, None) is not None:
            params[key] = value

    helper = Helper(params)
    run(helper)


if __name__ == '__main__':
    exp_name = 'mo_poison_001'
    iterations = 20
    search_space = {
        "momentum": 0.80333,
        "optimizer": 'Adam',
        "lr": 0.00072,
        # "label_noise": tune.uniform(0.0, 0.3),
        "decay": 0.0000048,
        "epochs": 15,
        "batch_size": 128,
        "random_seed": tune.grid_search(list(range(iterations))),
        # "drop_label_proportion": 0.95,
        "multi_objective_alpha": 0.99,
        "poisoning_proportion": 0.001,

        "wandb": {"project": f"random_seed", "group": exp_name, "monitor_gym": True}
    }
    config={}
    # runtime_env = RuntimeEnv(

    ray.init(address='ray://128.84.84.162:10001', runtime_env={"working_dir": "/home/eugene/irontorch",
                                                               'excludes': ['.git',
                                                                            '.data']},
             include_dashboard=True, dashboard_host='0.0.0.0')

    analysis = tune.run(tune_run, config=search_space, num_samples=1,
                        name=exp_name,
                        # scheduler=asha_scheduler,
                        # search_alg=optuna_search,
                        # resources_per_trial={'gpu': 1, 'cpu': 2},
                        loggers=[WandbLogger],
                        resources_per_trial=tune.PlacementGroupFactory([{"CPU": 4, "GPU": 1}]),
                        log_to_file=False,
                        fail_fast=True,
                        keep_checkpoints_num=1,
                        # sync_to_driver=False,
                        # metric='multi_objective',
                        # mode='max'
                        )

    print(
        "Best hyperparameters for accuracy found were: ",
        analysis.get_best_config("accuracy", "min"),
    )
    print(
        "Best hyperparameters for backdoor_accuracy found were: ",
        analysis.get_best_config("backdoor_accuracy", "max"),
    )