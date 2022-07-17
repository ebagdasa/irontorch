# import argparse
# import numpy as np
# from collections import defaultdict
# import ray
# from ray import tune
# from ray_training import tune_run, process_stage_1, process_stage_2
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='Tuning')
#     parser.add_argument('--random_seed', default=None, type=int)
#     parser.add_argument('--backdoor_label', default=None, type=int)
#     parser.add_argument('--poisoning_proportion', default=None, type=float)
#     parser.add_argument('--skip_stage3',  action='store_true')
#     parser.add_argument('--sub_exp_name', default=None, type=str)
#
#     args = parser.parse_args()
#
#     ray.init(address='ray://128.84.84.8:10001',
#              runtime_env={"working_dir": "/home/eugene/irontorch",
#                           'excludes': ['.git', '.data']},
#              include_dashboard=True, dashboard_host='0.0.0.0')
#     backdoor_cover_percentage = 0.01
#     search_alg = 'optuna'
#     exp_name = f'cifar_{search_alg}_it1'
#     if args.random_seed is None and args.backdoor_label is None:
#         # stage 1
#         group_name = f'stage1_{args.sub_exp_name}'
#         max_iterations = 50
#         full_exp_name = f'{exp_name}_{group_name}'
#         print(f'Running stage 1: {full_exp_name}')
#         search_space = {
#             'wandb_name': exp_name,
#             'group': group_name,
#             'random_seed': tune.choice(list(range(0, 50))),
#             'backdoor_label': tune.choice(list(range(0, 10))),
#             'epochs': 2,
#             'backdoor_cover_percentage': backdoor_cover_percentage,
#             'search_alg': None,
#             'poisoning_proportion': 0,
#             'file_path': '/home/eugene/irontorch/configs/cifar10_params.yaml',
#             'max_iterations': max_iterations
#         }
#         stage_1_results = tune_run(full_exp_name, search_space, resume=False)
#         backdoor_label, random_seed = process_stage_1(stage_1_results)
#         print(
#             f'Finished stage 1: backdoor_label: {backdoor_label} and random_seed: {random_seed}')
#     else:
#         print(
#             f'Skipping stage 1: reusing backdoor_label: {args.backdoor_label} and random_seed: {args.random_seed}')
#         backdoor_label = args.backdoor_label
#         random_seed = args.random_seed
#
#     if args.poisoning_proportion is None:
#         # stage 2
#         max_iterations = 40
#         group_name = f'stage2_{args.sub_exp_name}'
#         full_exp_name = f'{exp_name}_{group_name}'
#         print(f'Running stage 2: {full_exp_name}')
#         search_space = {
#             'wandb_name': exp_name,
#             'group': group_name,
#             'random_seed': random_seed,
#             'backdoor_label': backdoor_label,
#             'backdoor_cover_percentage': backdoor_cover_percentage,
#             'epochs': 30,
#             'search_alg': None,
#             'poisoning_proportion': tune.grid_search(list(np.arange(0, 50, 2))),
#             'file_path': '/home/eugene/irontorch/configs/cifar10_params.yaml',
#             'max_iterations': 1
#         }
#         stage_2_results = tune_run(full_exp_name, search_space, resume=False)
#         poisoning_proportion = process_stage_2(stage_2_results)
#         print(f'Finished stage 2: poisoning proportion: {poisoning_proportion}')
#     else:
#         print(f'Skipping stage 2: reusing poisoning_proportion: {args.poisoning_proportion}')
#         poisoning_proportion = args.poisoning_proportion
#
#     # stage 3
#     if not args.skip_stage3:
#         search_alg = 'optuna'
#         group_name = f'stage3_{args.sub_exp_name}'
#         metric_name = 'multi'
#         max_iterations = 100
#         full_exp_name = f'{exp_name}_{group_name}'
#         print(f'Running stage 3: {full_exp_name}')
#         search_space = {
#             "metric_name": metric_name,
#             'wandb_name': exp_name,
#             "group": group_name,
#             "optimizer": tune.choice(['SGD', 'Adam']),
#             "scheduler": False,
#             'random_seed': random_seed,
#             'backdoor_label': backdoor_label,
#             'backdoor_cover_percentage': backdoor_cover_percentage,
#             "poisoning_proportion": poisoning_proportion,
#             "grace_period": 2,
#             "lr": tune.qloguniform(1e-4, 2e-1, 1e-5, base=10),
#             "momentum": tune.quniform(0.1, 1.0, 0.05),
#             "decay": tune.qloguniform(1e-7, 1e-3, 1e-7, base=10),
#             "epochs": 30,
#             "batch_size": tune.choice([64, 128, 256]),
#             # "drop_label_proportion": 0.95,
#             "multi_objective_alpha": 0.95,
#             "search_alg": search_alg,
#             "grad_sigma": tune.qloguniform(1e-6, 1e-1, 1e-6, base=10),
#             "grad_clip": tune.qloguniform(1, 32, 1, base=2),
#             "label_noise": tune.quniform(0.0, 0.7, 0.01),
#             "file_path": '/home/eugene/irontorch/configs/cifar10_params.yaml',
#             "max_iterations": max_iterations
#         }
#
#         stage_4_results = tune_run(full_exp_name, search_space, resume=False)
#         print('Finished stage 3 tuning.')
#
#         # stage 4
#         group_name = f'stage4_{args.sub_exp_name}'
#         full_exp_name = f'{exp_name}_{group_name}'
#         print(f'Running stage 4: {full_exp_name}')
#         config = stage_4_results.get_best_config("multi_objective", "max")
#         print(config)
#         config['group'] = group_name
#         config['poisoning_proportion'] = tune.grid_search(list(np.arange(0, 50, 2)))
#         config['max_iterations'] = 1
#         config['search_alg'] = None
#         tune_run(full_exp_name, config)
#
