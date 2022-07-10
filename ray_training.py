import argparse
from ray.tune.integration.wandb import WandbLogger, WandbLoggerCallback
from ray.runtime_env import RuntimeEnv
from ray.tune.stopper import MaximumIterationStopper
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
    def wrapper(*args, **kwargs):
        logging.disable(logging.DEBUG)
        result = func(*args, **kwargs)
        logging.disable(logging.NOTSET)
        return result

    return wrapper


def run(config):
    with open(config['file_path']) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in config.items():
        if params.get(key, None) is not None:
            params[key] = value

    hlpr = Helper(params)
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        logging.disable(logging.DEBUG)
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        metrics = test(hlpr, hlpr.task.model, backdoor=False, epoch=epoch)
        drop_class = hlpr.task.metrics['accuracy'].get_value()[
            '_Accuracy_Drop_5']
        backdoor_metrics = test(hlpr, hlpr.task.model, backdoor=True,
                                epoch=epoch)
        main_obj = metrics[hlpr.params.multi_objective_metric]
        back_obj = backdoor_metrics[hlpr.params.multi_objective_metric]
        alpha = hlpr.params.multi_objective_alpha
        multi_obj = alpha * main_obj - (1 - alpha) * back_obj
        tune.report(accuracy=main_obj, drop_class=drop_class,
                    backdoor_accuracy=back_obj,
                    multi_objective=multi_obj, epoch=epoch,
                    poisoning_proportion=config['poisoning_proportion']
                    )


def tune_run(exp_name, search_space, resume=False):
    """
    Tune the model and return the best model.
    :param exp_name:
    :param search_space:
    :return:
    """
    callbacks = [WandbLoggerCallback(exp_name, group=search_space.get('group', None),
                                     excludes=["time_since_restore",
                                               "training_iteration",
                                               "warmup_time",
                                               "iterations_since_restore",
                                               "time_this_iter_s",
                                               "time_total_s",
                                               "timestamp",
                                               "timesteps_since_restore"])]
    name = search_space['name']
    if name == 'so':
        optuna_search = OptunaSearch(metric="accuracy", mode="max")
        asha_scheduler = ASHAScheduler(time_attr='epoch', metric='accuracy',
                                       mode='max', max_t=search_space['epochs'],
                                       grace_period=search_space['grace_period'],
                                       reduction_factor=4)
    elif name == 'mo':
        optuna_search = OptunaSearch(metric="multi_objective", mode="max")
        asha_scheduler = ASHAScheduler(time_attr='epoch', metric='multi_objective',
                                       mode='max', max_t=search_space['epochs'],
                                       grace_period=search_space['grace_period'],
                                       reduction_factor=4)
    elif name == 'multi' and search_space['search_alg'] == 'optuna':
        optuna_search = OptunaSearch(metric=["accuracy", "backdoor_accuracy"],
                                     mode=["max", "min"])
        asha_scheduler = None
    else:
        optuna_search = None
        asha_scheduler = None

    if search_space['search_alg'] == 'optuna':
        asha_scheduler = None
    else:
        optuna_search = None
    analysis = tune.run(run, config=search_space, num_samples=search_space['max_iterations'],
                    name=exp_name,
                    search_alg=optuna_search,
                    scheduler=asha_scheduler,
                    resources_per_trial=tune.PlacementGroupFactory(
                        [{"CPU": 4, "GPU": 1}]),
                    log_to_file=True,
                    fail_fast=True,
                    callbacks=callbacks,
                    keep_checkpoints_num=1,
                    resume=resume
                    )
    print(
        "Best hyperparameters for accuracy found were: ",
        analysis.get_best_config("accuracy", "min"),
    )
    print(
        "Best hyperparameters for backdoor_accuracy found were: ",
        analysis.get_best_config("backdoor_accuracy", "max"),
    )

    return analysis


if __name__ == '__main__':

    ray.init(address='ray://128.84.84.8:10001',
             runtime_env={"working_dir": "/home/eugene/irontorch",
                          'excludes': ['.git', '.data']},
             include_dashboard=True, dashboard_host='0.0.0.0')

    for name in ['multi']:
        poisoning_proportion = 50
        search_alg = 'optuna'
        exp_name = f'mnist_{search_alg}_{name}_p{poisoning_proportion}'
        max_iterations = 50
        search_space = {
            'name': 'multi',
            'group': 'p15',
            'random_seed': tune.choice(list(range(1, 50))),
             'optimizer': 'Adam',
             'lr': 9.211724509411425e-05,
             'momentum': 0.65,
             'grace_period': 2,
             'decay': 5.7183260054281126e-05,
             'epochs': 30,
             'batch_size': 32,
             'grad_sigma': 1.9652434554749976e-05,
             'grad_clip': 10.0,
             'label_noise': 0.25,
             'multi_objective_alpha': 0.97,
             'search_alg': 'optuna',
             'poisoning_proportion': 15,
             'file_path': '/home/eugene/irontorch/configs/mnist_params.yaml',
             'max_iterations': 50
        }
        # search_space = {
        #     "name": name,
        #     "optimizer": tune.choice(['SGD', 'Adam']),
        #     "lr": tune.qloguniform(1e-5, 2e-1, 1e-5),
        #     "momentum": tune.quniform(0.5, 0.95, 0.05),
        #     "grace_period": 2,
        #     "group": "search_for_hyperparameters",
        #     "decay": tune.qloguniform(1e-7, 1e-3, 1e-7, base=10),
        #     "epochs": 30,
        #     "batch_size": tune.choice([32, 64, 128, 256, 512]),
        #     # "transform_sharpness": tune.loguniform(1e-4, 1, 10),
        #     # "transform_erase": tune.loguniform(1e-4, 1, 10),
        #     "grad_sigma": tune.qloguniform(1e-5, 1e-1, 5e-6, base=10),
        #     "grad_clip": tune.quniform(1, 10, 1),
        #     "label_noise": tune.quniform(0.0, 0.5, 0.05),
        #     # "drop_label_proportion": 0.95,
        #     "multi_objective_alpha": 0.97,
        #     "search_alg": search_alg,
        #     "poisoning_proportion": poisoning_proportion, #tune.qloguniform(2, 50000, 1, base=10),
        #     "file_path": '/home/eugene/irontorch/configs/mnist_params.yaml',
        #     "max_iterations": max_iterations
        #
        # }
        analysis = tune_run(exp_name, search_space, resume=False)
        # print('Finished tuning')
        # config = analysis.get_best_config("multi_objective", "max")
        # print(config)
        # config['poisoning_proportion'] = tune.choice(list(range(0, 500, 5)))
        # config['max_iterations'] = 100
        # config['group'] = 'robust'
        # config['search_alg'] = None
        # tune_run(exp_name, config)

