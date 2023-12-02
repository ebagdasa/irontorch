# IronTorch project

Please install requirements.txt. To run the experiments we used `ray_training.
py`. For example:

```python
python ray_training.py --task mnist --sub_exp_name test_exp \ 
--random_seed 6 --backdoor_label 9 --metric_name multi_objective \
--search_alg optuna --synthesizer Primitive --backdoor_cover_percentage 0.05 \
--stage3_max_iterations 360  --multi_objective_alpha 0.9
```
The approach requires ray==1.13.0