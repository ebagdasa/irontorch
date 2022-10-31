import argparse
import os
from copy import deepcopy

from ray.tune import ExperimentAnalysis
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.sigopt import SigOptSearch
from ray.tune.schedulers.pbt import PopulationBasedTraining
from collections import defaultdict

from ray.tune.suggest.zoopt import ZOOptSearch

from helper import Helper
from training import train, test
import yaml
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, \
    AsyncHyperBandScheduler
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
        if hlpr.params.final_test_only:
            continue

        metrics = test(hlpr, hlpr.task.model, backdoor=False, epoch=epoch, val=hlpr.params.val_only)
        drop_class = hlpr.task.metrics['accuracy'].get_value().get('_Accuracy_Drop_5', 0)
        backdoor_metrics = test(hlpr, hlpr.task.model, backdoor=True,
                                epoch=epoch, val=hlpr.params.val_only, synthesizer=hlpr.params.main_synthesizer)
        main_obj = metrics[hlpr.params.multi_objective_metric]
        back_accuracy = backdoor_metrics[hlpr.params.multi_objective_metric]
        back_obj = 100 - back_accuracy
        alpha = hlpr.params.multi_objective_alpha
        multi_obj = alpha * main_obj - (1 - alpha) * back_accuracy
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
    if hlpr.params.final_test_only:
        results_metrics = dict()
        results_metrics['poisoning_proportion'] = params['poisoning_proportion']
        main_obj = test(hlpr, hlpr.task.model, backdoor=False, epoch=hlpr.params.epochs,
                        val=hlpr.params.val_only)['accuracy']
        results_metrics['drop_class'] = hlpr.task.metrics['accuracy'].get_value().get('_Accuracy_Drop_5', 0)
        results_metrics['accuracy'] = main_obj
        results_metrics['epoch'] = hlpr.params.epochs
        for i, synthesizer in enumerate(hlpr.params.synthesizers):
            back_accuracy = test(hlpr, hlpr.task.model, backdoor=True, epoch=hlpr.params.epochs,
                            val=hlpr.params.val_only,
                            synthesizer=synthesizer)['accuracy']
            back_obj = 100 - back_accuracy
            results_metrics[f'backdoor_{synthesizer}'] = back_accuracy
            if i == 0:
                alpha = hlpr.params.multi_objective_alpha
                results_metrics['backdoor_accuracy'] = back_accuracy
                results_metrics['backdoor_error'] = back_obj
                results_metrics['multi_objective'] = alpha * main_obj + (1 - alpha) * back_obj
                results_metrics['anti_obj'] = alpha * main_obj + (1 - alpha) * back_accuracy
        tune.report(**results_metrics)


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
    max_epoch = params['epochs'] if isinstance(params['epochs'], int) else params['epochs'].upper
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
    callbacks = None
    metric_name = params.get('metric_name', None)
    if metric_name == 'multi':
        alg_metrics = ["accuracy", "backdoor_error"]
        alg_modes = ["max", "min"]
        scheduler_metrics = "multi_objective"
        scheduler_modes = "max"
    else:
        alg_metrics = metric_name
        alg_modes = "max"
        scheduler_metrics = alg_metrics
        scheduler_modes = alg_modes

    scheduler = None
    search_algo = None
    if params['search_alg'] == 'OptunaSearch':
        search_algo = OptunaSearch(metric=alg_metrics, mode=alg_modes)
    elif params['search_alg'] == 'SigOptSearch':
        search_algo = SigOptSearch(metric=alg_metrics, mode=alg_modes)
    elif params['search_alg'] == 'HyperOptSearch':
        search_algo = HyperOptSearch(metric=alg_metrics, mode=alg_modes)
    elif params['search_alg'] == 'TuneBOHB':
        search_algo = TuneBOHB(metric=alg_metrics, mode=alg_modes)
    elif params['search_alg'] == 'ZOOptSearch':
        search_algo = ZOOptSearch(metric=alg_metrics, mode=alg_modes)
    elif params['search_alg'] == 'BayesOptSearch':
        search_algo = BayesOptSearch(metric=alg_metrics, mode=alg_modes)
    elif params['search_alg'] is None:
        print("No search algorithm specified")
    else:
        raise ValueError('Invalid search algorithm')

    if params['search_scheduler'] == 'ASHAScheduler':
        scheduler = ASHAScheduler(time_attr='epoch', metric=scheduler_metrics,
                                       mode=scheduler_modes, max_t=max_epoch,
                                       grace_period=params['grace_period'],
                                       reduction_factor=4)
    elif params['search_scheduler'] == 'HyperBandForBOHB':
        scheduler = HyperBandForBOHB(metric=scheduler_metrics, mode=scheduler_modes,
                                          max_t=max_epoch, reduction_factor=4)
    elif params['search_scheduler'] == 'MedianStoppingRule':
        scheduler = MedianStoppingRule(metric=scheduler_metrics, mode=scheduler_modes,
                                            time_attr='epoch', grace_period=params['grace_period'])
    elif params['search_scheduler'] == 'AsyncHyperBandScheduler':
        scheduler = AsyncHyperBandScheduler(metric=scheduler_metrics, mode=scheduler_modes,
                                                 time_attr='epoch', max_t=max_epoch,
                                                 grace_period=params['grace_period'],
                                                 reduction_factor=4)
    elif params['search_scheduler'] is None:
        print("No scheduler algorithm specified")
    else:
        raise ValueError('Invalid search algorithm')

    analysis = tune.run(run, config=params, num_samples=params['max_iterations'],
                        name=exp_name,
                        search_alg=search_algo,
                        scheduler=scheduler,
                        resources_per_trial=tune.PlacementGroupFactory(
                            [{"CPU": 4, "GPU": 1}]),
                        log_to_file=True,
                        fail_fast=False,
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
        if x.is_finished():
            pp[x.config['poisoning_proportion']] = x.last_result['backdoor_error']
    min_error = min(pp.values()) + 40 # 10 is a small offset to avoid the case where the minimum is 0
    for pois_prop, error in sorted(pp.items(), key=lambda x: x[0]):
        if error <= min_error:
            return pois_prop
    raise ValueError("Didn't work")


def add_secret_config(old_config):
    old_config['main_synthesizer'] = old_config['synthesizers'][0]
    old_config['synthesizers'].append('Secret')
    old_config['backdoor_labels']['Secret'] = 1

    return old_config


def add_imbalance(old_config):
    old_config['drop_label'] = 5
    old_config['drop_label_proportion'] = 0.9

    return old_config


def parametrize_mnist(old_config):
    old_config['out_channels1'] = tune.lograndint(8, 128, base=2)
    old_config['out_channels2'] = tune.lograndint(8, 128, base=2)
    old_config['kernel_size1'] = tune.randint(1, 6)
    old_config['kernel_size2'] = tune.randint(1, 6)
    old_config['strides1'] = 1
    old_config['strides2'] = 1
    old_config['dropout1'] = tune.uniform(0, 0.99)
    old_config['dropout2'] = tune.uniform(0, 0.99)
    old_config['fc1'] = tune.randint(8, 512)
    old_config['max_pool'] = tune.randint(1, 3)
    old_config['activation'] = tune.choice(['relu', 'tanh', 'sigmoid',
                                            'elu', 'leaky_relu', 'selu'])

    # old_config['batch_clip'] = False
    # old_config['grad_clip'] = 1000
    # old_config['grad_sigma'] = 0.0

    return old_config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Ray Tuning')
    parser.add_argument('--random_seed', default=None, type=int)
    parser.add_argument('--backdoor_label', default=None, type=int)
    parser.add_argument('--poisoning_proportion', default=None, type=float)
    parser.add_argument('--load_stage1', default=None, type=str)
    parser.add_argument('--load_stage3', default=None, type=str)
    parser.add_argument('--sub_exp_name', default=None, type=str)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--task', default='mnist', type=str)
    parser.add_argument('--search_alg', default=None, type=str)
    parser.add_argument('--search_scheduler', default=None, type=str)
    parser.add_argument('--metric_name', required=True, type=str)
    parser.add_argument('--backdoor_cover_percentage', default=0.1, type=float)
    parser.add_argument('--synthesizer', default='Primitive', type=str)
    parser.add_argument('--stage4_run_name', default=None, type=str)
    parser.add_argument('--skip_stage4', action='store_true')
    parser.add_argument('--stage3_max_iterations', default=306, type=int)
    parser.add_argument('--stage4_multi_backdoor', action='store_true')
    parser.add_argument('--final_test_only', action='store_true')
    parser.add_argument('--multi_objective_alpha', default=0.9, type=float)
    parser.add_argument('--add_secret_config', action='store_true')
    parser.add_argument('--add_imbalance', action='store_true')


    args = parser.parse_args()

    if args.local:
        ray.init()
    else:
        ray.init(address='ray://128.84.80.37:10001',
             runtime_env={"working_dir": "/home/eugene/irontorch",
                          'excludes': ['.git', '.data'],
                          "env_vars": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
                          },
             include_dashboard=True, dashboard_host='0.0.0.0')
    print(f'RUNNING {args.task} config.')
    proportions_min = {'SinglePixel': 0, 'Dynamic': 0, 'Pattern': 0, 'Primitive': 0,
                       'Complex': 4, 'Clean': 4}
    batch_size = tune.choice([32, 64, 128, 256, 512])
    if args.task == 'mnist':
        epochs = 5
        proportion_to_test = [5*i for i in range(36)] #np.unique(np.logspace(0, 10, num=80, base=2, dtype=np.int32)).tolist()
        proportions = {'SinglePixel': 12, 'Dynamic': 14, 'Pattern': 12, 'Primitive': 12,
                       'Complex': 14, 'Clean': 15}
    elif args.task == 'cifar10':
        epochs = 1
        proportion_to_test = np.unique(np.logspace(0, 10, num=27, base=2, dtype=np.int32)).tolist()
        proportions = {'SinglePixel': 15, 'Dynamic': 15, 'Pattern': 10, 'Primitive': 10,
                       'Complex': 15, 'Clean': 12}
        proportions_min = {'SinglePixel': 1, 'Dynamic': 1, 'Pattern': 0, 'Primitive': 0,
                           'Complex': 5, 'Clean': 1}
    elif args.task == 'cifar100':
        epochs = 10
        proportion_to_test = np.unique(np.logspace(0, 10, num=27, base=2, dtype=np.int32)).tolist()
        proportions = {'SinglePixel': 11, 'Dynamic': 11, 'Pattern': 9, 'Primitive': 9,
                       'Complex': 12, 'Clean': 9}
    elif args.task == 'celeba':
        epochs = 1
        batch_size = tune.choice([32, 64, 128, ])
        proportion_to_test = np.unique(np.logspace(0, 12, num=40, base=2, dtype=np.int32)).tolist()
        proportions = {'SinglePixel': 16, 'Dynamic': 16, 'Pattern': 12, 'Primitive': 12,
                       'Complex': 15, 'Clean': 16}
    elif args.task == 'imagenet':
        epochs = 16
        proportion_to_test = np.unique(np.logspace(6, 10, num=12, base=2, dtype=np.int32)).tolist()
        proportions = {'SinglePixel': 16, 'Dynamic': 16, 'Pattern': 14, 'Primitive': 8,
                       'Complex': 17, 'Clean': 10}
        proportions_min = {'SinglePixel': 1, 'Dynamic': 1, 'Pattern': 1, 'Primitive': 3,
                           'Complex': 1, 'Clean': 1}
        batch_size = tune.choice([128, 256, 512, 1024])
    else:
        raise ValueError(f'Unknown task {args.task}')

    file_path = f'/home/eugene/irontorch/configs/{args.task}_params.yaml'
    metric_name = args.metric_name
    exp_name = f'{args.task}_hypersearch'
    poisoning_proportion = args.poisoning_proportion
    backdoor_label = args.backdoor_label
    random_seed = args.random_seed
    group_name = f'stage3_{args.sub_exp_name}'
    max_iterations = args.stage3_max_iterations
    full_exp_name = f'{exp_name}_{group_name}'
    print(f'Running stage 3: {full_exp_name}')
    print(f'AAA{args.synthesizer}: {backdoor_label}')

    for searcher in [None, 'TuneBOHB', 'HyperOptSearch', 'ZOOptSearch', 'OptunaSearch', 'NevergradSearch', 'SigOptSearch', 'BayesOptSearch']:
        for scheduler in [None, 'ASHAScheduler', 'HyperBandForBOHB', 'MedianStoppingRule', 'AsyncHyperBandScheduler']:

            full_exp_name = f'{exp_name}_{group_name}_{searcher}_{scheduler}'
            search_space = {
                'synthesizers': [args.synthesizer],
                'backdoor_labels': {args.synthesizer: backdoor_label},
                "metric_name": metric_name,
                'wandb_name': exp_name,
                "optimizer": tune.choice(['SGD', 'Adam','Adadelta']),
                "lr": tune.qloguniform(1e-5, 2, 1e-5),
                "scheduler": tune.choice(['StepLR', 'MultiStepLR', 'CosineAnnealingLR']),
                "momentum": tune.quniform(0.1, 0.9, 0.1),
                "grace_period": 2,
                "stage": 3,
                "group": group_name,
                "decay": tune.qloguniform(1e-7, 1e-3, 1e-7, base=10),
                "epochs": epochs,
                'random_seed': random_seed,
                "backdoor_cover_percentage": args.backdoor_cover_percentage,
                "batch_size": batch_size,

                # "transform_sharpness": tune.loguniform(1e-4, 1, 10),
                'batch_clip': True,
                # "transform_erase": tune.loguniform(1e-4, 0.4, 10),
                "grad_sigma": tune.qloguniform(1e-5, 1e-1, 5e-6, base=10),
                "grad_clip": tune.quniform(1, 50, 0.1),
                "label_noise": tune.quniform(0.0, 0.9, 0.01),
                # "cifar_model_l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
                # "cifar_model_l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
                "multi_objective_alpha": args.multi_objective_alpha,
                "search_alg": searcher,
                "search_scheduler": scheduler,
                "poisoning_proportion": poisoning_proportion,
                "file_path": file_path,
                "max_iterations": max_iterations,
                'val_only': True,
                'backdoor': True,
                'final_test_only': args.final_test_only
            }
            if args.task == 'mnist':
                search_space = parametrize_mnist(search_space)
            if args.add_imbalance:
                search_space = add_imbalance(search_space)
            if args.add_secret_config:
                search_space = add_secret_config(search_space)
            print(search_space)
            if os.path.exists(f"/home/eugene/ray_results/{full_exp_name}"):
                print('ATTEMPTING TO RESUME STAGE 3')
                print(f'Loading from {full_exp_name}')
                stage_3_results = tune_run(full_exp_name, search_space, resume=True)
            else:
                stage_3_results = tune_run(full_exp_name, search_space, resume=False)
            config = stage_3_results.get_best_config("multi_objective", "max")
            print('Finished stage 3 tuning.')
