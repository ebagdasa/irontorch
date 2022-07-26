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


def run(params):

    for key, value in params.items():
        if params.get(key, None) is not None:
            params[key] = value

    hlpr = Helper(params)
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        logging.disable(logging.DEBUG)
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        metrics = test(hlpr, hlpr.task.model, backdoor=False, epoch=epoch, val=hlpr.params.val_only)
        drop_class = hlpr.task.metrics['accuracy'].get_value()[
            '_Accuracy_Drop_5']
        backdoor_metrics = test(hlpr, hlpr.task.model, backdoor=True,
                                epoch=epoch, val=hlpr.params.val_only)
        main_obj = metrics[hlpr.params.multi_objective_metric]
        back_accuracy = backdoor_metrics[hlpr.params.multi_objective_metric]
        back_obj = 100 - back_accuracy
        alpha = hlpr.params.multi_objective_alpha
        multi_obj = alpha * main_obj + (1 - alpha) * back_obj
        anti_obj = alpha * main_obj + (1 - alpha) * back_accuracy
        lr = hlpr.task.scheduler.get_last_lr()[
            0] if hlpr.task.scheduler is not None else hlpr.params.lr
        tune.report(accuracy=main_obj, drop_class=drop_class,
                    backdoor_accuracy=back_accuracy,
                    anti_obj=anti_obj,
                    loss=metrics['loss'],
                    backdoor_loss=backdoor_metrics['loss'],
                    backdoor_error=back_obj,
                    multi_objective=multi_obj, epoch=epoch,
                    poisoning_proportion=params['poisoning_proportion'],
                    learning_rate=lr
                    )


def tune_run(exp_name, search_space, resume=False):
    """
    Tune the model and return the best model.
    :param exp_name:
    :param search_space:
    :return:
    """
    with open(search_space['file_path']) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params.update(search_space)

    callbacks = [WandbLoggerCallback(params.get('wandb_name', exp_name),
                                     group=params.get('group', None),
                                     excludes=["time_since_restore",
                                               "training_iteration",
                                               "warmup_time",
                                               "iterations_since_restore",
                                               "time_this_iter_s",
                                               "time_total_s",
                                               "timestamp",
                                               "timesteps_since_restore"])]
    metric_name = params.get('metric_name', None)
    if metric_name == 'multi':
        optuna_search = OptunaSearch(metric=["accuracy", "backdoor_error"],
                                     mode=["max", "min"])
        asha_scheduler = ASHAScheduler(time_attr='epoch', metric='multi_objective',
                                       mode='max', max_t=params['epochs'],
                                       grace_period=params['grace_period'],
                                       reduction_factor=4)
    else:
        optuna_search = OptunaSearch(metric=metric_name, mode="max")
        asha_scheduler = ASHAScheduler(time_attr='epoch', metric=metric_name,
                                       mode='max', max_t=params['epochs'],
                                       grace_period=params['grace_period'],
                                       reduction_factor=4)
    if params['search_alg'] == 'optuna':
        asha_scheduler = None
    elif params['search_alg'] == 'asha':
        optuna_search = None
    elif params['search_alg'] == 'both':
        pass
    else:
        raise ValueError('Invalid search algorithm')

    analysis = tune.run(run, config=params, num_samples=params['max_iterations'],
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

    parser = argparse.ArgumentParser(description='Ray Tuning')
    parser.add_argument('--random_seed', default=None, type=int)
    parser.add_argument('--backdoor_label', default=None, type=int)
    parser.add_argument('--poisoning_proportion', default=None, type=float)
    parser.add_argument('--load_stage1', default=None, type=str)
    parser.add_argument('--load_stage3', default=None, type=str)
    parser.add_argument('--sub_exp_name', default=None, type=str)
    parser.add_argument('--task', default='mnist', type=str)
    parser.add_argument('--search_alg', required=True, type=str)
    parser.add_argument('--metric_name', required=True, type=str)
    parser.add_argument('--backdoor_cover_percentage', default=None, type=float)
    parser.add_argument('--synthesizer', default='Pattern', type=str)
    parser.add_argument('--stage4_run_name', default=None, type=str)
    parser.add_argument('--backdoor_dynamic_position', default=False, type=bool)
    parser.add_argument('--stage3_max_iterations', default=306, type=int)

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
        proportion_to_test = np.unique(np.logspace(0, 10, num=40, base=2, dtype=np.int32)).tolist()
    elif args.task == 'cifar10':
        epochs = 10
        proportion_to_test = np.unique(np.logspace(3, 9, num=40, base=2, dtype=np.int32)).tolist()
    else:
        raise ValueError(f'Unknown task {args.task}')

    if args.backdoor_dynamic_position:
        proportion_to_test = np.unique(np.logspace(0, 15, num=40, base=2, dtype=np.int32)).tolist()


    file_path = f'/home/eugene/irontorch/configs/{args.task}_params.yaml'
    search_alg = args.search_alg
    metric_name = args.metric_name
    exp_name = f'{args.task}_hypersearch'
    if args.random_seed is None and args.backdoor_label is None:
        # stage 0
        group_name = f'stage0_{args.sub_exp_name}'
        max_iterations = 50
        full_exp_name = f'{exp_name}_{group_name}'
        print(f'Running stage 0: {full_exp_name}')
        search_space = {
            'synthesizer': args.synthesizer,
            'wandb_name': exp_name,
            'group': group_name,
            'random_seed': tune.choice(list(range(0, 50))),
            'backdoor_label': tune.choice(list(range(0, 10))),
            'epochs': 1,
            'batch_clip': False,
            "stage": 0,
            'backdoor_cover_percentage': args.backdoor_cover_percentage,
            'search_alg': None,
            'poisoning_proportion': 0,
            'file_path': file_path,
            'max_iterations': max_iterations,
            'backdoor_dynamic_position': args.backdoor_dynamic_position,
            # "cifar_model_l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            # "cifar_model_l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        }
        stage_1_results = tune_run(full_exp_name, search_space, resume=False)
        backdoor_label, random_seed = process_stage_1(stage_1_results)
        print(f'Finished stage 0: backdoor_label: {backdoor_label} and random_seed: {random_seed}')
        with open(f"/home/eugene/ray_results/{full_exp_name}/results.txt", 'a') as f:
            f.write(f'backdoor_label: {backdoor_label}' + '\n')
            f.write(f'random_seed: {random_seed}' + '\n')
    else:
        print(
            f'Skipping stage 0: reusing backdoor_label: {args.backdoor_label} and random_seed: {args.random_seed}')
        backdoor_label = args.backdoor_label
        random_seed = args.random_seed

    if args.load_stage1 is None:
        # stage 1
        group_name = f'stage1_{args.sub_exp_name}'
        max_iterations = 54
        full_exp_name = f'{exp_name}_{group_name}'
        print(f'Running stage 1: {full_exp_name}')
        search_space = {
            'synthesizer': args.synthesizer,
            'backdoor_cover_percentage': args.backdoor_cover_percentage,
            "metric_name": "accuracy",
            'wandb_name': exp_name,
            "optimizer": tune.choice(['SGD', 'Adam', 'Adadelta']),
            "lr": tune.qloguniform(1e-5, 2, 1e-5),
            "scheduler": tune.choice(['StepLR', 'MultiStepLR', 'CosineAnnealingLR']),
            "momentum": tune.quniform(0.1, 0.9, 0.1),
            "group": group_name,
            "grace_period": 2,
            "stage": 1,
            "decay": tune.qloguniform(1e-7, 1e-3, 1e-7, base=10),
            "epochs": epochs,
            'random_seed': random_seed,
            'backdoor_label': backdoor_label,
            "batch_size": tune.choice([32, 64, 128, 256, 512]),
            'batch_clip': False,
            "search_alg": search_alg,
            "poisoning_proportion": 0.0,
            "file_path": file_path,
            "max_iterations": max_iterations,
            'backdoor_dynamic_position': args.backdoor_dynamic_position,
            # "cifar_model_l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            # "cifar_model_l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        }
        stage_1_results = tune_run(full_exp_name, search_space, resume=False)
        stage_1_config = stage_1_results.get_best_config(metric='accuracy', mode='max')
    else:
        print(f'Skipping stage 1')
        try:
            path = f"/home/eugene/ray_results/{args.load_stage1}/"
            stage_1_results = ExperimentAnalysis(path)
            stage_1_config = stage_1_results.get_best_config(metric='accuracy', mode='max')
            print(f'Loaded stage 1 config: {stage_1_config}')
        except Exception as e:
            print(f'Error loading stage 1 results: {e}. using empty config')
            stage_1_config = {}

    if args.poisoning_proportion is None:
        # stage 2
        group_name = f'stage2_{args.sub_exp_name}'
        full_exp_name = f'{exp_name}_{group_name}'
        print(f'Running stage 2: {full_exp_name}')
        search_space = {
            'synthesizer': args.synthesizer,
            'wandb_name': exp_name,
            'metric_name': None,
            'group': group_name,
            'random_seed': random_seed,
            'backdoor_label': backdoor_label,
            'backdoor_cover_percentage': args.backdoor_cover_percentage,
            'epochs': epochs,
            "stage": 2,
            'batch_clip': False,
            'search_alg': None,
            'poisoning_proportion': tune.grid_search(proportion_to_test),
            'file_path': file_path,
            'max_iterations': 1,
            'backdoor_dynamic_position': args.backdoor_dynamic_position,
            # "cifar_model_l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            # "cifar_model_l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        }
        stage_1_config.update(search_space)
        print(f'New stage 2 config: {stage_1_config}')
        stage_2_results = tune_run(full_exp_name, stage_1_config, resume=False)
        poisoning_proportion = process_stage_2(stage_2_results)
        print(f'Finished stage 2: poisoning proportion: {poisoning_proportion}')
        with open(f"/home/eugene/ray_results/{full_exp_name}/results.txt", 'a') as f:
            f.write(f'poisoning_proportion: {poisoning_proportion}')
    else:
        print(f'Skipping stage 2: reusing poisoning_proportion: {args.poisoning_proportion}')
        poisoning_proportion = args.poisoning_proportion
    # stage 3
    if not args.load_stage3:
        group_name = f'stage3_{args.sub_exp_name}'
        max_iterations = args.stage3_max_iterations
        full_exp_name = f'{exp_name}_{group_name}'
        print(f'Running stage 3: {full_exp_name}')

        search_space = {
            'synthesizer': args.synthesizer,
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
            "backdoor_cover_percentage": args.backdoor_cover_percentage,
            "batch_size": tune.choice([32, 64, 128, 256, 512]),

            # "transform_sharpness": tune.loguniform(1e-4, 1, 10),
            'batch_clip': False,
            "transform_erase": tune.loguniform(1e-4, 1, 10),
            "grad_sigma": tune.qloguniform(1e-5, 1e-1, 5e-6, base=10),
            "grad_clip": tune.quniform(1, 10, 1),
            "label_noise": tune.quniform(0.0, 0.7, 0.02),
            # "cifar_model_l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            # "cifar_model_l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            # "drop_label_proportion": 0.95,
            "multi_objective_alpha": 0.9,
            "search_alg": search_alg,
            "poisoning_proportion": poisoning_proportion,
            "file_path": file_path,
            "max_iterations": max_iterations,
            'backdoor_dynamic_position': args.backdoor_dynamic_position
        }
        stage_3_results = tune_run(full_exp_name, search_space, resume=False)
        config = stage_3_results.get_best_config("multi_objective", "max")
        print('Finished stage 3 tuning.')
    else:
        path = f"/home/eugene/ray_results/{args.load_stage3}/"
        print(f'Skipping stage 3: Loading results from {path}')
        stage_3_results = ExperimentAnalysis(path)

    # stage 4
    if args.stage4_run_name is None:
        config = stage_3_results.get_best_config("multi_objective", "max")
    else:
        config = stage_3_results.results[args.stage4_run_name]['config']
        print(f'Loaded run: {args.stage4_run_name}')
    print(config)

    def update_conf(config, part):
        group_name = f'stage4_{args.sub_exp_name}_p{part}'
        full_exp_name = f'{exp_name}_{group_name}'
        print(f'Running stage 4: {full_exp_name}. Part {part}')
        config['wandb_name'] = exp_name
        config['group'] = group_name
        config['synthesizer'] = args.synthesizer
        config['backdoor_cover_percentage'] = args.backdoor_cover_percentage
        config['stage'] = f'4.{part}'
        config['poisoning_proportion'] = tune.grid_search(proportion_to_test)
        config['backdoor_dynamic_position'] = args.backdoor_dynamic_position
        config['max_iterations'] = 1
        config['search_alg'] = None
        config['val_only'] = True

        return full_exp_name, config


    full_exp_name, config = update_conf(config, 1)
    tune_run(full_exp_name, config)

    config = stage_3_results.get_best_config("accuracy", "max")
    full_exp_name, config = update_conf(config, 2)
    tune_run(full_exp_name, config)

    config = stage_3_results.get_best_config("anti_obj", "max")
    full_exp_name, config = update_conf(config, 3)
    tune_run(full_exp_name, config)
    