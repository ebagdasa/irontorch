import ray
from ray import tune
from ray_training import tune_run

if __name__ == '__main__':

    ray.init(address='ray://128.84.84.8:10001',
             runtime_env={"working_dir": "/home/eugene/irontorch",
                          'excludes': ['.git', '.data']},
             include_dashboard=True, dashboard_host='0.0.0.0')

    for _ in ['multi']:
        poisoning_proportion = 50
        search_alg = None #'optuna'
        exp_name = f'cifar_run_normal2'
        max_iterations = 80
        search_space = {
            "name": 'normal',
            "batch_size": 128,
            # "optimizer": tune.choice(['SGD', 'Adam']),
            # # "scheduler": tune.choice([True, False]),
            # "grace_period": 2,
            # "lr": tune.qloguniform(1e-5, 2e-1, 1e-5, base=10),
            # "momentum": tune.quniform(0.1, 1.0, 0.05),
            # "decay": tune.qloguniform(1e-7, 1e-3, 1e-7, base=10),
            "epochs": 20,
            # "batch_size": tune.choice([32, 64, 128, 256]),
            # # "drop_label_proportion": 0.95,
            # "multi_objective_alpha": 0.95,
            "search_alg": search_alg,
            # # "transform_sharpness": tune.quniform(0.0, 0.5, 0.01),
            # "transform_erase": tune.quniform(0.0, 1.0, 0.01),
            # "grad_sigma": tune.qloguniform(1e-6, 1e-1, 1e-6, base=10),
            # "grad_clip": tune.qloguniform(1, 32, 1, base=2),
            # "label_noise": tune.quniform(0.0, 0.5, 0.01),
            "poisoning_proportion": tune.randint(5, 60),
            "file_path": '/home/eugene/irontorch/configs/cifar10_params.yaml',
            "max_iterations": max_iterations
        }
        tune_run(exp_name, search_space, resume=False)
