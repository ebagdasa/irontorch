import argparse
from copy import deepcopy

from ray.tune import ExperimentAnalysis
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.sigopt import SigOptSearch
from ray.tune.schedulers.pbt import PopulationBasedTraining
from collections import defaultdict

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
    if params['search_alg'] == 'optuna':
        search_algo = OptunaSearch(metric=alg_metrics, mode=alg_modes)
    elif params['search_alg'] == 'sigopt':
        search_algo = SigOptSearch(metric=alg_metrics, mode=alg_modes)
    elif params['search_alg'] == 'hyperopt':
        search_algo = HyperOptSearch(metric=alg_metrics, mode=alg_modes)
    elif params['search_alg'] is None:
        print("No search algorithm specified")
    else:
        raise ValueError('Invalid search algorithm')

    if params['search_scheduler'] == 'asha':
        scheduler = ASHAScheduler(time_attr='epoch', metric=scheduler_metrics,
                                       mode=scheduler_modes, max_t=max_epoch,
                                       grace_period=params['grace_period'],
                                       reduction_factor=4)
    elif params['search_scheduler'] == 'PopulationBasedTraining':
        scheduler = PopulationBasedTraining(time_attr='epoch', metric=scheduler_metrics,
                                       mode=scheduler_modes)
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
        if x.is_finished():
            pp[x.config['poisoning_proportion']] = x.last_result['backdoor_error']
    min_error = min(pp.values()) + 10 # 10 is a small offset to avoid the case where the minimum is 0
    for pois_prop, error in sorted(pp.items(), key=lambda x: x[0]):
        if error <= min_error:
            return pois_prop
    raise ValueError("Didn't work")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Ray Tuning')
    parser.add_argument('--random_seed', default=None, type=int)
    parser.add_argument('--backdoor_label', default=None, type=int)
    parser.add_argument('--poisoning_proportion', default=None, type=float)
    parser.add_argument('--load_stage1', default=None, type=str)
    parser.add_argument('--load_stage3', default=None, type=str)
    parser.add_argument('--sub_exp_name', default=None, type=str)
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


    args = parser.parse_args()

    ray.init(address='ray://128.84.80.37:10001',
             runtime_env={"working_dir": "/home/eugene/irontorch",
                          'excludes': ['.git', '.data'],
                          "env_vars": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
                          },
             include_dashboard=True, dashboard_host='0.0.0.0')
    print(f'RUNNING {args.task} config.')
    proportions_min = {'SinglePixel': 0, 'Dynamic': 0, 'Pattern': 0, 'Primitive': 0,
                       'Complex': 4, 'Clean': 4}
    batch_size = tune.choice([32, 64, 16, 48, 8])
    if args.task == 'mnist_fed':
        epochs = 20
        proportion_to_test = [5*i for i in range(36)] #np.unique(np.logspace(0, 10, num=80, base=2, dtype=np.int32)).tolist()
        proportions = {'SinglePixel': 10, 'Dynamic': 14, 'Pattern': 9, 'Primitive': 9,
                       'Complex': 14, 'Clean': 13}
    elif args.task == 'cifar10':
        epochs = 10
        proportion_to_test = np.unique(np.logspace(0, 10, num=27, base=2, dtype=np.int32)).tolist()
        proportions = {'SinglePixel': 11, 'Dynamic': 11, 'Pattern': 9, 'Primitive': 9,
                       'Complex': 12, 'Clean': 12}
    elif args.task == 'cifar100':
        epochs = 10
        proportion_to_test = np.unique(np.logspace(0, 10, num=27, base=2, dtype=np.int32)).tolist()
        proportions = {'SinglePixel': 11, 'Dynamic': 11, 'Pattern': 9, 'Primitive': 9,
                       'Complex': 12, 'Clean': 9}
    elif args.task == 'celeba':
        epochs = 5
        proportion_to_test = np.unique(np.logspace(0, 10, num=40, base=2, dtype=np.int32)).tolist()
        proportions = {'SinglePixel': 16, 'Dynamic': 16, 'Pattern': 12, 'Primitive': 12,
                       'Complex': 15, 'Clean': 16}
    elif args.task == 'imagenet':
        epochs = 1
        proportion_to_test = np.unique(np.logspace(1, 14, num=18, base=2, dtype=np.int32)).tolist()
        proportions = {'SinglePixel': 16, 'Dynamic': 16, 'Pattern': 14, 'Primitive': 14,
                       'Complex': 17, 'Clean': 10}
        proportions_min = {'SinglePixel': 1, 'Dynamic': 1, 'Pattern': 1, 'Primitive': 3,
                           'Complex': 1, 'Clean': 1}
        batch_size = tune.choice([32, 64, 128,])
    else:
        raise ValueError(f'Unknown task {args.task}')

    file_path = f'/home/eugene/irontorch/configs/{args.task}_params.yaml'
    metric_name = args.metric_name
    exp_name = f'{args.task}_hypersearch'
    if args.random_seed is None and args.backdoor_label is None:
        # stage 0
        group_name = f'stage0_{args.sub_exp_name}'
        max_iterations = 54
        full_exp_name = f'{exp_name}_{group_name}'
        print(f'Running stage 0: {full_exp_name}')
        search_space = {
            'synthesizers': [args.synthesizer],
            'wandb_name': exp_name,
            'group': group_name,
            'random_seed': tune.choice(list(range(0, 50))),
            'backdoor_labels': {args.synthesizer: tune.choice(list(range(0, 10)))},
            'epochs': 1,
            'batch_clip': False,
            "stage": 0,
            'backdoor_cover_percentage': args.backdoor_cover_percentage,
            'search_alg': None,
            'search_scheduler': None,
            'poisoning_proportion': 0,
            'file_path': file_path,
            'max_iterations': max_iterations,
            'backdoor': True,
            'final_test_only': args.final_test_only
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
            'synthesizers': [args.synthesizer],
            'backdoor_labels': {args.synthesizer: backdoor_label},
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
            "batch_size": batch_size,
            'batch_clip': False,

            "fl_local_epochs": tune.randint(1, 5),
            "fl_eta": tune.qloguniform(1e-5, 10, 1e-5),
            "fl_no_models": tune.randint(5, 50),
            "fl_diff_privacy": False,

            "label_noise": 0,
            "transform_erase": 0,
            "transform_sharpness": 0,
            "search_alg": args.search_alg,
            "search_scheduler": args.search_scheduler,
            "poisoning_proportion": 0,
            "file_path": file_path,
            "max_iterations": max_iterations,
            'val_only': True,
            'backdoor': True,
            'final_test_only': args.final_test_only
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

    if (args.poisoning_proportion is None) and (args.load_stage3 is None):
        # stage 2
        group_name = f'stage2_{args.sub_exp_name}'
        full_exp_name = f'{exp_name}_{group_name}'
        print(f'Running stage 2: {full_exp_name}')
        search_space = {
            'synthesizers': [args.synthesizer],
            'backdoor_labels': {args.synthesizer: backdoor_label},
            'wandb_name': exp_name,
            'metric_name': None,
            'group': group_name,
            'random_seed': random_seed,
            'backdoor_cover_percentage': args.backdoor_cover_percentage,
            'epochs': epochs,
            "stage": 2,
            'batch_clip': False,
            'search_alg': None,
            'search_scheduler': None,
            'poisoning_proportion': tune.grid_search(proportion_to_test),
            'file_path': file_path,
            'max_iterations': 1,
            'val_only': True,
            'backdoor': True,
            'final_test_only': args.final_test_only
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
        print(f'AAA{args.synthesizer}: {backdoor_label}')

        search_space = {
            'synthesizers': [args.synthesizer],
            'backdoor_labels': {args.synthesizer: backdoor_label},
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
            "epochs": tune.randint(epochs-4, epochs+4),
            'random_seed': random_seed,
            "backdoor_cover_percentage": args.backdoor_cover_percentage,
            "batch_size": batch_size,

            "fl_local_epochs": tune.randint(1, 5),
            "fl_eta": tune.qloguniform(1e-5, 10, 1e-5),
            "fl_no_models": tune.randint(5, 50),
            "fl_dp_noise": tune.qloguniform(1e-5, 1e-1, 5e-6, base=10),
            "fl_dp_clip": tune.quniform(1, 10, 1),
            "fl_diff_privacy": True,

            "multi_objective_alpha": args.multi_objective_alpha,
            "search_alg": args.search_alg,
            "search_scheduler": args.search_scheduler,
            "poisoning_proportion": poisoning_proportion,
            "file_path": file_path,
            "max_iterations": max_iterations,
            'val_only': True,
            'backdoor': True,
            'final_test_only': args.final_test_only
        }
        print(search_space)
        stage_3_results = tune_run(full_exp_name, search_space, resume=False)
        config = stage_3_results.get_best_config("multi_objective", "max")
        print('Finished stage 3 tuning.')
    else:
        path = f"/home/eugene/ray_results/{args.load_stage3}/"
        print(f'Skipping stage 3: Loading results from {path}')
        stage_3_results = ExperimentAnalysis(path)

    # stage 4
    if args.skip_stage4:
        raise ValueError('Skipping stage 4')


    if args.stage4_run_name is None:
        stage_3_config = stage_3_results.get_best_config("multi_objective", "max")
    else:
        stage_3_config = stage_3_results.results[args.stage4_run_name]['config']
        print(f'Loaded run: {args.stage4_run_name}')
    print(stage_3_config)




    def update_conf(config, part, synthesizer):
        config = deepcopy(config)
        print(f'Running stage 4, part {part} synthesize: {synthesizer}')
        if config.get('synthesizer', None):
            config.pop('synthesizer')
            config.pop('backdoor_label')
        proportion = np.unique(np.logspace(proportions_min[synthesizer], proportions[synthesizer], num=20, base=2, dtype=np.int32, endpoint=True)).tolist()
        proportion = [0] + proportion
        group_name = f'stage4_{args.sub_exp_name}_p{part}_{synthesizer}'
        full_exp_name = f'{exp_name}_{group_name}'
        print(f'Running stage 4: {full_exp_name}. Part {part}')
        config['wandb_name'] = exp_name
        config['group'] = group_name
        config['backdoor_cover_percentage'] = args.backdoor_cover_percentage
        config['stage'] = f'4.{part}'
        config['poisoning_proportion'] = tune.grid_search(proportion)
        config['max_iterations'] = 1
        config['search_alg'] = None
        config['search_scheduler'] = None
        config['synthesizers'] = [args.synthesizer]
        config['epochs'] = epochs

        config['synthesizers'] = [synthesizer]
        config['backdoor_labels'] = {synthesizer: backdoor_label}

        config['main_synthesizer'] = 'Pattern'
        config['split_val_test_ratio'] = 0.4
        config['final_test_only'] = True
        config['val_only'] = False
        return full_exp_name, config


    if args.stage4_multi_backdoor:
        synthesizers = ['Primitive', 'SinglePixel', 'Dynamic', 'Pattern', 'Complex', 'Clean']
    else:
        synthesizers = [args.synthesizer]

    for synthesizer in synthesizers:
        full_exp_name, config = update_conf(stage_3_config, 1, synthesizer)
        tune_run(full_exp_name, config)

        # config = stage_3_results.get_best_config("accuracy", "max")
        # full_exp_name, config = update_conf(config, 2)
        # tune_run(full_exp_name, config)

        # config = stage_3_results.get_best_config("anti_obj", "max")
        # full_exp_name, config = update_conf(config, 3)
        # tune_run(full_exp_name, config)

        if len(stage_1_config) != 0:
            full_exp_name, config = update_conf(stage_1_config, 4, synthesizer)
            tune_run(full_exp_name, config)
