import argparse
import numpy as np
from collections import defaultdict
import ray
from ray import tune
from ray_training import tune_run

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tuning')
    parser.add_argument('--random_seed', default=None, type=int)
    parser.add_argument('--backdoor_label', default=None, type=int)
    parser.add_argument('--poisoning_proportion', default=None, type=float)

    args = parser.parse_args()

    ray.init(address='ray://128.84.84.8:10001',
             runtime_env={"working_dir": "/home/eugene/irontorch",
                          'excludes': ['.git', '.data']},
             include_dashboard=True, dashboard_host='0.0.0.0')

    search_alg = 'optuna'
    exp_name = f'cifar_{search_alg}_it1'
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
            'backdoor_cover_percentage': 0.01,
            'search_alg': None,
            'poisoning_proportion': 0,
            'file_path': '/home/eugene/irontorch/configs/cifar10_params.yaml',
            'max_iterations': max_iterations
        }
        stage_1_results = tune_run(full_exp_name, search_space, resume=False)
        label = defaultdict(list)
        for x in stage_1_results.trials:
            if x.is_finished():
                label[x.config['backdoor_label']].append(
                    (x.conf['random_seed'], x.last_result['backdoor_accuracy']))
        min_var_arg = np.argmin([np.var([z for _, z in label[x]]) for x in range(0, 10)])
        backdoor_label = min_var_arg
        random_seed = sorted(label[min_var_arg], key=lambda x: x[1])[0][0]
        print(
            f'Finished stage 1: backdoor_label: {args.backdoor_label} and random_seed: {args.random_seed}')
    else:
        print(
            f'Skipping stage 1: reusing backdoor_label: {args.backdoor_label} and random_seed: {args.random_seed}')
        backdoor_label = args.backdoor_label
        random_seed = args.random_seed

    if args.poisoning_proportion is None:
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
            'epochs': 30,
            'search_alg': None,
            'poisoning_proportion': tune.lograndint(1, 10000, base=10),
            'file_path': '/home/eugene/irontorch/configs/mnist_params.yaml',
            'max_iterations': max_iterations
        }
        stage_2_results = tune_run(full_exp_name, search_space, resume=False)
        pp = dict()
        for x in stage_2_results.trials:
            if x.is_finished() and x.last_result['epoch'] == x.config['epochs']:
                pp[x.config['poisoning_proportion']] = x.last_result['backdoor_error'] < 20
        z = sorted(pp.items(), key=lambda x: x[0])
        zz = [z[i][0] for i in range(1, len(z) - 2) if z[i][1] and z[i + 1][1]]
        poisoning_proportion = min(zz)
        print(f'Finished stage 2: poisoning proportion: {poisoning_proportion}')
    else:
        print(f'Skipping stage 2: reusing poisoning_proportion: {args.poisoning_proportion}')
        poisoning_proportion = args.poisoning_proportion

    # stage 3
    print('Running stage 3')
    search_alg = 'optuna'
    group_name = 'stage3'
    metric_name = 'multi'
    max_iterations = 500
    full_exp_name = f'{exp_name}_{group_name}'
    search_space = {
        "metric_name": metric_name,
        'wandb_name': exp_name,
        "group": group_name,
        "optimizer": 'SGD', #tune.choice(['SGD', 'Adam']),
        "scheduler": tune.choice([True, False]),
        "grace_period": 2,
        "lr": tune.qloguniform(1e-5, 2e-1, 1e-5, base=10),
        "momentum": tune.quniform(0.1, 1.0, 0.05),
        "decay": tune.qloguniform(1e-7, 1e-3, 1e-7, base=10),
        "epochs": 90,
        "batch_size": tune.choice([32, 64, 128, 256]),
        # "drop_label_proportion": 0.95,
        "multi_objective_alpha": 0.95,
        "search_alg": search_alg,
        "grad_sigma": tune.qloguniform(1e-6, 1e-1, 1e-6, base=10),
        "grad_clip": tune.qloguniform(1, 32, 1, base=2),
        "label_noise": tune.quniform(0.0, 0.5, 0.01),
        "poisoning_proportion": poisoning_proportion,
        "file_path": '/home/eugene/irontorch/configs/cifar10_params.yaml',
        "max_iterations": max_iterations
    }

    analysis = tune_run(full_exp_name, search_space, resume=False)
    print('Finished stage 3 tuning.')

    # stage 4
    print('Running stage 4')
    group_name = 'stage4'
    full_exp_name = f'{exp_name}_{group_name}'
    config = analysis.get_best_config("multi_objective", "max")
    print(config)
    config['group'] = group_name
    config['poisoning_proportion'] = tune.lograndint(1, 10000, base=10)
    config['max_iterations'] = 100
    config['search_alg'] = None
    tune_run(full_exp_name, config)

