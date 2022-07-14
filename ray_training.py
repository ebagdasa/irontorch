import argparse
from ray.tune.integration.wandb import WandbLogger, WandbLoggerCallback
from ray.runtime_env import RuntimeEnv
from ray.tune.stopper import MaximumIterationStopper
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.optuna import OptunaSearch
from collections import defaultdict
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
    metrics = test(hlpr, hlpr.task.model, backdoor=False, epoch=0)
    drop_class = hlpr.task.metrics['accuracy'].get_value()[
        '_Accuracy_Drop_5']
    backdoor_metrics = test(hlpr, hlpr.task.model, backdoor=True,
                            epoch=0)
    main_obj = metrics[hlpr.params.multi_objective_metric]
    back_obj = backdoor_metrics[hlpr.params.multi_objective_metric]
    alpha = hlpr.params.multi_objective_alpha
    multi_obj = alpha * main_obj - (1 - alpha) * back_obj
    tune.report(accuracy=main_obj, drop_class=drop_class,
                backdoor_accuracy=back_obj,
                multi_objective=multi_obj, epoch=0,
                poisoning_proportion=config['poisoning_proportion']
                )
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
    callbacks = [WandbLoggerCallback(search_space.get('wandb_name', exp_name),
                                     group=search_space.get('group', None),
                                     excludes=["time_since_restore",
                                               "training_iteration",
                                               "warmup_time",
                                               "iterations_since_restore",
                                               "time_this_iter_s",
                                               "time_total_s",
                                               "timestamp",
                                               "timesteps_since_restore"])]
    name = search_space.get('name', None)
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
    parser = argparse.ArgumentParser(description='Tuning')
    parser.add_argument('--random_seed', default=None)
    parser.add_argument('--backdoor_label', default=None)
    parser.add_argument('--backdoor_proportion', default=None)

    args = parser.parse_args()

    ray.init(address='ray://128.84.84.8:10001',
             runtime_env={"working_dir": "/home/eugene/irontorch",
                          'excludes': ['.git', '.data']},
             include_dashboard=True, dashboard_host='0.0.0.0')
    search_alg = 'optuna'
    exp_name = f'mnist_{search_alg}_it1'
    if args.random_seed is None and args.backdoor_label is None:
        # stage 1
        print('Running stage 1')
        group_name = 'stage1'
        max_iterations = 50
        full_exp_name = f'{exp_name}_{group_name}'
        search_space = {
            'wandb_name': exp_name,
            'group': group_name,
            'random_seed': tune.choice(list(range(0, 50))),
            'backdoor_label': tune.choice(list(range(0, 10))),
            'epochs': 2,
            'backdoor_cover_percentage': 0.1,
            'search_alg': None,
            'poisoning_proportion': 0,
            'file_path': '/home/eugene/irontorch/configs/mnist_params.yaml',
            'max_iterations': max_iterations
        }
        stage_1_results = tune_run(full_exp_name, search_space, resume=False)
        label = defaultdict(list)
        for x in stage_1_results.trials:
            if x.is_finished():
                label[x.config['backdoor_label']].append( (x.conf['random_seed'], x.last_result['backdoor_accuracy']))
        min_var_arg = np.argmin([np.var([z for _, z in label[x]]) for x in range(0, 10)])
        backdoor_label = min_var_arg
        random_seed = sorted(label[min_var_arg], key=lambda x: x[1])[0][0]
    else:
        print(f'Reusing backdoor_label: {args.backdoor_label} and random_seed: {args.random_seed}')
        backdoor_label = args.backdoor_label
        random_seed = args.random_seed

    if args.backdoor_proportion is None:
        # stage 2
        print('Running stage 2')
        max_iterations = 50
        group_name = 'stage2'
        full_exp_name = f'{exp_name}_{group_name}'
        search_space = {
            'wandb_name': exp_name,
            'group': group_name,
            'random_seed': random_seed,
            'backdoor_label': backdoor_label,
            'epochs': 2,
            'backdoor_cover_percentage': 0.1,
            'search_alg': None,
            'poisoning_proportion': tune.lograndint(1, 10000, base=10),
            'file_path': '/home/eugene/irontorch/configs/mnist_params.yaml',
            'max_iterations': max_iterations
        }
        stage_2_results = tune_run(full_exp_name, search_space, resume=False)

    #
    #
    # for name in ['multi']:
    #     poisoning_proportion = 0.00000001
    #     search_alg = 'optuna'
    #     full_exp_name = f'mnist_{search_alg}_{name}_p15'
    #     max_iterations = 200
    #     search_space = {
    #         'name': 'multi',
    #         'group': 'p0.5_labels',
    #         'random_seed': tune.choice(list(range(0, max_iterations//10))),
    #         'backdoor_label': tune.choice(list(range(0, 10))),
    #          'epochs': 2,
    #          'batch_size': 32,
    #          'backdoor_cover_percentage': 0.5,
    #          'search_alg': None,
    #          'poisoning_proportion': poisoning_proportion,
    #          'file_path': '/home/eugene/irontorch/configs/mnist_params.yaml',
    #          'max_iterations': max_iterations
    #     }
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
        # analysis = tune_run(full_exp_name, search_space, resume=False)
        # print('Finished tuning')
        # config = analysis.get_best_config("multi_objective", "max")
        # print(config)
        # config['poisoning_proportion'] = tune.choice(list(range(0, 500, 5)))
        # config['max_iterations'] = 100
        # config['group'] = 'robust'
        # config['search_alg'] = None
        # tune_run(exp_name, config)

