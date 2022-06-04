import argparse
from ray.tune.integration.wandb import WandbLogger
from ray.runtime_env import RuntimeEnv

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
        tune.report(accuracy=metrics['accuracy'], backdoor_accuracy=backdoor_metrics['accuracy'])
        # hlpr.plot_confusion_matrix(backdoor=True, epoch=epoch)
        # hlpr.save_model(hlpr.task.model, epoch, metrics['accuracy'])
        # if hlpr.params.multi_objective_metric is not None:
        #     main_obj = metrics[hlpr.params.multi_objective_metric]
        #     back_obj = backdoor_metrics[hlpr.params.multi_objective_metric]
        #     alpha = hlpr.params.multi_objective_alpha
        #     multi_obj = alpha * main_obj - (1 - alpha) * back_obj
        #     hlpr.report_dict(dict_report={'multi_objective': multi_obj}, step=epoch)


def tune_run(config):
    with open('/home/eugene/backdoors/configs/mnist_params.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in config.items():
        if params.get(key, None) is not None:
            params[key] = value
    helper = Helper(params)
    run(helper)


if __name__ == '__main__':
    search_space = {
        "momentum": tune.uniform(0.1, 0.9),
        "optimizer": tune.choice(['Adam', 'SGD']),
        "lr": tune.loguniform(1e-4, 1e-1),
        "wandb": {"project": "rayTune", "monitor_gym": True}
    }
    runtime_env = RuntimeEnv(
        conda='pt',
        working_dir="/home/eugene/irontorch",
        # py_modules=[Helper, test, train]

    )
    ray.init(address='ray://128.84.84.162:10001', runtime_env=runtime_env)

    analysis = tune.run(tune_run, config=search_space, num_samples=10,
                        # resources_per_trial={'gpu': 1, 'cpu': 2},
                        loggers=[WandbLogger],
                        resources_per_trial=tune.PlacementGroupFactory([{"CPU": 2, "GPU": 1}])
                        )