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
logger = logging.getLogger('logger')

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
    with open('/home/eugene/irontorch/configs/cifar10_params.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in config.items():
        if params.get(key, None) is not None:
            params[key] = value
            logger.error(f'Updating {key} with {value}')
    helper = Helper(params)
    run(helper)
    # main_obj, back_obj, multi_obj = list(), list(), list()
    # for x in range(10):
    #     params['random_seed'] = x
    #     helper = Helper(params)
    #     acc, back_acc, mo = run(helper)
    #     main_obj.append(acc)
    #     back_obj.append(back_acc)
    #     multi_obj.append(mo)
    #     if acc <= 80 or back_acc >= 90:
    #         tune.report(accuracy=np.mean(main_obj),
    #                     backdoor_accuracy=np.mean(back_obj),
    #                     multi_objective=np.mean(multi_obj),
    #                     acc_std=np.std(main_obj),
    #                     back_std=np.std(back_obj)
    #                     )
    #         return
    # tune.report(accuracy=np.mean(main_obj),
    #             backdoor_accuracy=np.mean(back_obj),
    #             multi_objective=np.mean(multi_obj),
    #             acc_std=np.std(main_obj),
    #             back_std=np.std(back_obj))


if __name__ == '__main__':
    exp_name = 'cifar_30'
    search_space = {
        "optimizer": 'SGD',
        "lr": tune.loguniform(1e-7, 3e-1, 10),
        # "scheduler": tune.choice([False, True]),
        "momentum": tune.uniform(0, 1),
        # "label_noise": tune.uniform(0.0, 0.3),
        "decay": tune.loguniform(1e-7, 1e-1, 10),
        "epochs": 100,
        "batch_size": tune.grid_search([32, 64, 128, 256]),
        # "drop_label_proportion": 0.95,
        "multi_objective_alpha": 0.99,
        "poisoning_proportion": 30,
        "wandb": {"project": f"rayTune_{exp_name}", "monitor_gym": True}
    }
    asha_scheduler = ASHAScheduler(
        time_attr='epoch',
        metric='multi_objective',
        mode='max',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
    )
    config={}
    # runtime_env = RuntimeEnv(
    #     conda='pt',
    #     working_dir="/home/eugene/irontorch",
    #     # py_modules=[Helper, test, train]
    #
    # )
    # hyperopt_search = HyperOptSearch(search_space, metric="multi_objective", mode="max")
    # optuna_search = OptunaSearch(metric="accuracy", mode="max")
    optuna_search = OptunaSearch(metric=["accuracy", "backdoor_accuracy"], mode=["max", "min"])

    ray.init(address='ray://128.84.84.162:10001', runtime_env={"working_dir": "/home/eugene/irontorch",
                                                               'excludes': ['.git',
                                                                            '.data']},
             include_dashboard=True, dashboard_host='0.0.0.0')

    analysis = tune.run(tune_run, config=search_space, num_samples=100,
                        name=exp_name,
                        scheduler=asha_scheduler,
                        # search_alg=optuna_search,
                        # resources_per_trial={'gpu': 1, 'cpu': 2},
                        loggers=[WandbLogger],
                        resources_per_trial=tune.PlacementGroupFactory([{"CPU": 4, "GPU": 1}]),
                        log_to_file=True,
                        fail_fast=True,
                        # max_failures=1,
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