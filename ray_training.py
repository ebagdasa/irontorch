import argparse

from ray.tune import ExperimentAnalysis
from ray.tune.integration.wandb import WandbLogger, WandbLoggerCallback
from ray.runtime_env import RuntimeEnv
from ray.tune.stopper import MaximumIterationStopper
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.optuna import OptunaSearch
from collections import defaultdict
import training
import json

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
        back_obj = 100 - backdoor_metrics[hlpr.params.multi_objective_metric]
        alpha = hlpr.params.multi_objective_alpha
        multi_obj = alpha * main_obj + (1 - alpha) * back_obj
        lr = hlpr.task.scheduler.get_last_lr()[0] if hlpr.task.scheduler is not None else hlpr.params.lr
        tune.report(accuracy=main_obj, drop_class=drop_class,
                    backdoor_accuracy=backdoor_metrics[hlpr.params.multi_objective_metric],
                    loss=metrics['loss'],
                    backdoor_loss=backdoor_metrics['loss'],
                    backdoor_error=back_obj,
                    multi_objective=multi_obj, epoch=epoch,
                    poisoning_proportion=config['poisoning_proportion'],
                    learning_rate=lr
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
    metric_name = search_space.get('metric_name', None)
    if metric_name == 'so':
        optuna_search = OptunaSearch(metric="accuracy", mode="max")
        asha_scheduler = ASHAScheduler(time_attr='epoch', metric='accuracy',
                                       mode='max', max_t=search_space['epochs'],
                                       grace_period=search_space['grace_period'],
                                       reduction_factor=4)
    elif metric_name == 'mo':
        optuna_search = OptunaSearch(metric="multi_objective", mode="max")
        asha_scheduler = ASHAScheduler(time_attr='epoch', metric='multi_objective',
                                       mode='max', max_t=search_space['epochs'],
                                       grace_period=search_space['grace_period'],
                                       reduction_factor=4)
    elif metric_name == 'multi' and search_space['search_alg'] == 'optuna':
        optuna_search = OptunaSearch(metric=["accuracy", "backdoor_error"],
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


def process_stage_1(analysis):
    label = defaultdict(list)
    for x in analysis.trials:
        if x.is_finished():
            label[x.config['backdoor_label']].append(
                (x.config['random_seed'], x.last_result['backdoor_error']))
    min_var_arg = np.argmin([np.var([z for _, z in label[x]]) for x in range(0, 10)])
    backdoor_label = min_var_arg
    random_seed = sorted(label[min_var_arg], key=lambda x: x[1])[-1][0]

    return backdoor_label, random_seed


def process_stage_2(analysis):
    pp = dict()
    for x in analysis.trials:
        if x.is_finished() and x.last_result['epoch'] == x.config['epochs']:
            pp[x.config['poisoning_proportion']] = x.last_result['backdoor_error'] < 50
    z = sorted(pp.items(), key=lambda x: x[0])
    zz = [z[i][0] for i in range(1, len(z) - 2) if z[i][1] and z[i + 1][1]]
    return min(zz)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tuning MNIST')
    parser.add_argument('--random_seed', default=None, type=int)
    parser.add_argument('--backdoor_label', default=None, type=int)
    parser.add_argument('--poisoning_proportion', default=None, type=float)
    parser.add_argument('--load_stage3',  default=None, type=str)
    parser.add_argument('--sub_exp_name', default=None, type=str)
    parser.add_argument('--task', default='mnist', type=str)
    parser.add_argument('--search_alg', default='optuna', type=str)
    parser.add_argument('--backdoor_cover_percentage', default=0.01, type=float)

    args = parser.parse_args()

    ray.init(address='ray://128.84.84.8:10001',
             runtime_env={"working_dir": "/home/eugene/irontorch",
                          'excludes': ['.git', '.data'],
                          "env_vars": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
                          },
             include_dashboard=True, dashboard_host='0.0.0.0')
    print(f'RUNNING {args.task} config.')
    if args.task == 'mnist':
        epochs = 5
    elif args.task == 'cifar10':
        epochs = 10
    else:
        raise ValueError(f'Unknown task {args.task}')

    file_path = f'/home/eugene/irontorch/configs/{args.task}_params.yaml'
    search_alg = args.search_alg
    exp_name = f'{args.task}_{search_alg}'
    if args.random_seed is None and args.backdoor_label is None:
        # stage 1
        group_name = f'stage1_{args.sub_exp_name}'
        max_iterations = 50
        full_exp_name = f'{exp_name}_{group_name}'
        print(f'Running stage 1: {full_exp_name}')
        search_space = {
            'wandb_name': exp_name,
            'group': group_name,
            'random_seed': tune.choice(list(range(0, 50))),
            'backdoor_label': tune.choice(list(range(0, 10))),
            'epochs': 1,
            'batch_clip': False,
            "stage": 1,
            'backdoor_cover_percentage': args.backdoor_cover_percentage,
            'search_alg': None,
            'poisoning_proportion': 0,
            'file_path': file_path,
            'max_iterations': max_iterations
        }
        stage_1_results = tune_run(full_exp_name, search_space, resume=False)
        backdoor_label, random_seed = process_stage_1(stage_1_results)
        print(f'Finished stage 1: backdoor_label: {backdoor_label} and random_seed: {random_seed}')
        with open(f"/home/eugene/ray_results/{full_exp_name}/results.txt", 'a') as f:
            f.write(f'backdoor_label: {backdoor_label}' + '\n')
            f.write(f'random_seed: {random_seed}' + '\n')
    else:
        print(f'Skipping stage 1: reusing backdoor_label: {args.backdoor_label} and random_seed: {args.random_seed}')
        backdoor_label = args.backdoor_label
        random_seed = args.random_seed

    if args.poisoning_proportion is None:
        # stage 2
        group_name = f'stage2_{args.sub_exp_name}'
        full_exp_name = f'{exp_name}_{group_name}'
        print(f'Running stage 2: {full_exp_name}')
        search_space = {
            'wandb_name': exp_name,
            'group': group_name,
            'random_seed': random_seed,
            'backdoor_label': backdoor_label,
            'backdoor_cover_percentage': args.backdoor_cover_percentage,
            'epochs': epochs,
            "stage": 2,
            'batch_clip': False,
            'search_alg': None,
            'poisoning_proportion': tune.grid_search(list(np.arange(0, 400, 4))),
            'file_path': file_path,
            'max_iterations': 1
        }
        stage_2_results = tune_run(full_exp_name, search_space, resume=False)
        poisoning_proportion = process_stage_2(stage_2_results)
        print(f'Finished stage 2: poisoning proportion: {poisoning_proportion}')
        with open(f"/home/eugene/ray_results/{full_exp_name}/results.txt", 'a') as f:
            f.write(f'poisoning_proportion: {poisoning_proportion}')
    else:
        print(f'Skipping stage 2: reusing poisoning_proportion: {args.poisoning_proportion}')
        poisoning_proportion = args.poisoning_proportion
    # stage 3
    if not args.load_stage3:
        search_alg = 'optuna'
        group_name = f'stage3_{args.sub_exp_name}'
        metric_name = 'multi'
        max_iterations = 300
        full_exp_name = f'{exp_name}_{group_name}'
        print(f'Running stage 3: {full_exp_name}')

        search_space = {
            "metric_name": metric_name,
            'wandb_name': exp_name,
            "optimizer": tune.choice(['SGD', 'Adam', 'Adadelta']),
            "lr": tune.qloguniform(1e-5, 2, 1e-5),
            "scheduler": tune.choice(['StepLR', 'MultiStepLR', 'CosineAnnealingLR']),
            "momentum": tune.quniform(0.1, 0.9, 0.1),
            "grace_period": 2,
            "stage": 3,
            "group": group_name,
            "decay": tune.qloguniform(1e-7, 1e-3, 1e-7, base=10),
            "epochs": epochs,
            'random_seed': random_seed,
            'backdoor_label': backdoor_label,
            "batch_size": tune.choice([32, 64, 128, 256, 512]),
            'batch_clip': False,
            # "transform_sharpness": tune.loguniform(1e-4, 1, 10),
            # "transform_erase": tune.loguniform(1e-4, 1, 10),
            # "grad_sigma": tune.qloguniform(1e-5, 1e-1, 5e-6, base=10),
            # "grad_clip": tune.quniform(1, 10, 1),
            "label_noise": tune.quniform(0.0, 0.5, 0.05),
            # "drop_label_proportion": 0.95,
            "multi_objective_alpha": 0.97,
            "search_alg": search_alg,
            "poisoning_proportion": poisoning_proportion,
            "file_path": file_path,
            "max_iterations": max_iterations
        }
        stage_3_results = tune_run(full_exp_name, search_space, resume=False)
        config = stage_3_results.get_best_config("multi_objective", "max")
        with open(f"/home/eugene/ray_results/{full_exp_name}/results.txt", 'a') as f:
            json.dump(config, f)
        print('Finished stage 3 tuning.')
    else:
        path = f"/home/eugene/ray_results/{args.load_stage3}/"
        print(f'Skipping stage 3: Loading results from {path}')
        stage_3_results = ExperimentAnalysis(path)
        config = stage_3_results.get_best_config("multi_objective", "max")

    # stage 4
    group_name = f'stage4_{args.sub_exp_name}'
    full_exp_name = f'{exp_name}_{group_name}'
    print(f'Running stage 4: {full_exp_name}')


    print(config)
    config['group'] = group_name
    config['stage'] = 4
    config['poisoning_proportion'] = tune.grid_search(list(np.arange(0, 400, 4)))
    config['max_iterations'] = 1
    config['search_alg'] = None
    tune_run(full_exp_name, config)

