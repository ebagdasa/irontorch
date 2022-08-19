import argparse
import shutil
import torch
from datetime import datetime

import yaml
from prompt_toolkit import prompt
from tqdm import tqdm
from copy import deepcopy

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from losses.loss_functions import compute_normal_loss
from utils.utils import *

logger = logging.getLogger('logger')


def get_percentage(params, train_dataset, batch):
    attack_count = 0
    drop_label = 0
    for i, x in enumerate(batch.indices):
        if train_dataset.true_targets[x].item() != params.backdoor_labels[params.main_synthesizer] and \
              batch.aux[i].item() == 1:
            attack_count += 1
        if params.drop_label is not None and train_dataset.targets[x] == params.drop_label:
            drop_label += 1
    return 100.0 * attack_count/batch.indices.shape[0],  100.0 * drop_label/batch.indices.shape[0]


def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True):
    criterion = hlpr.task.criterion
    model.train()
    total_train = len(train_loader) if not hlpr.params.max_batch_id else hlpr.params.max_batch_id
    for i, data in tqdm(enumerate(train_loader), disable=True, total=total_train):
        batch = hlpr.task.get_batch(i, data)
        optimizer.zero_grad()
        if hlpr.params.label_noise:
            mask = (torch.rand(size=batch.labels.shape, device=hlpr.params.device) >= hlpr.params.label_noise)
            mask = mask.type(batch.labels.dtype)
            rand_labels = torch.randint_like(batch.labels, 0, len(hlpr.task.train_dataset.classes))
            batch.labels = mask * batch.labels + (1 - mask) * rand_labels
        outputs = model(batch.inputs)
        loss = criterion(outputs, batch.labels)
        loss = loss.mean()
        hlpr.params.running_losses['normal'].append(loss.item())
        attack_percent, drop_label = get_percentage(hlpr.params, hlpr.task.train_dataset, batch)
        hlpr.params.running_losses['attack_percent'].append(attack_percent)
        hlpr.params.running_losses['drop_label'].append(drop_label)
        if hlpr.params.saved_grads and hlpr.params.opacus:
            optimizer.batch_idx = i
            optimizer.data_accum[i] = batch.indices.detach().cpu()
            optimizer.label_accum[i] = batch.labels.detach().cpu()
            optimizer.aux[i] = batch.aux.detach().cpu()
            optimizer.loss_accum[i] = loss.detach().cpu()
        if loss.item() > 20:
            print('oh, high loss')
        loss.backward()
        if hlpr.params.batch_clip:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hlpr.params.grad_clip)
            hlpr.params.running_losses['total_norm'].append(total_norm.item())
            if hlpr.params.grad_sigma > 0.0:
                for param in model.parameters():
                    noised_layer = torch.FloatTensor(param.shape)
                    noised_layer = noised_layer.to(param.device)
                    noised_layer.normal_(mean=0, std=hlpr.params.grad_sigma)
                    param.grad.add_(noised_layer)
        optimizer.step()

        hlpr.report_training_losses_scales(i, epoch)
        if i*hlpr.params.batch_size >= hlpr.params.max_batch_id:
            break
    hlpr.task.scheduler_step()
    return


def test(hlpr: Helper, model, backdoor=False, epoch=None, val=False, synthesizer=None):
    model.eval()
    hlpr.task.reset_metrics()
    if backdoor is True and val is True:
        loader = hlpr.task.val_attack_loaders[synthesizer]
    elif backdoor is True and val is False:
        loader = hlpr.task.test_attack_loaders[synthesizer]
    elif backdoor is False and val is True:
        loader = hlpr.task.val_loader
    else:
        loader = hlpr.task.test_loader

    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), disable=True):
            batch = hlpr.task.get_batch(i, data)
            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    prefix = f'Backdoor_{synthesizer}' if backdoor else 'Normal'
    test_type = 'Val' if val else 'Test'
    metrics = hlpr.report_metrics(prefix=f'{test_type}_{prefix}', epoch=epoch)
    # metric = hlpr.task.report_metrics(epoch,
    #                          prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
    #                          tb_writer=hlpr.tb_writer,
    #                          tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return dict((name, metric.get_main_metric_value()) for name, metric in hlpr.task.metrics.items())


def run(hlpr):
    # acc = test(hlpr, hlpr.task.model, backdoor=True, epoch=0)
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        metrics = test(hlpr, hlpr.task.model, backdoor=False, epoch=epoch, val=True)
        hlpr.plot_confusion_matrix(backdoor=False, epoch=epoch)
        backdoor_metrics = dict()
        for synthesizer in hlpr.params.synthesizers:
            backdoor_metrics[synthesizer] = test(hlpr, hlpr.task.model, backdoor=True, epoch=epoch,
                                    val=True, synthesizer=synthesizer)
            hlpr.plot_confusion_matrix(backdoor=True, epoch=epoch)
        hlpr.save_model(hlpr.task.model, epoch, metrics['accuracy'])
        if hlpr.params.multi_objective_metric is not None:
            main_obj = metrics[hlpr.params.multi_objective_metric]
            back_obj = backdoor_metrics[hlpr.params.main_synthesizer][hlpr.params.multi_objective_metric]
            alpha = hlpr.params.multi_objective_alpha
            multi_obj = alpha * main_obj - (1 - alpha) * back_obj
            hlpr.report_dict(dict_report={'multi_objective': multi_obj}, step=epoch)

    metrics = test(hlpr, hlpr.task.model, backdoor=False, epoch=0, val=False)
    hlpr.plot_confusion_matrix(backdoor=False, epoch=0)
    for synthesizer in hlpr.params.synthesizers:
        backdoor_metrics = test(hlpr, hlpr.task.model, backdoor=True, epoch=0, val=False, synthesizer=synthesizer)
        hlpr.plot_confusion_matrix(backdoor=True, epoch=0)




def fl_run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, hlpr.task.model, backdoor=False)
        test(hlpr, hlpr.task.model, backdoor=True)

        hlpr.save_model(hlpr.task.model, epoch, metric)


def run_fl_round(hlpr, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        for local_epoch in range(hlpr.params.fl_local_epochs):
            if user.compromised:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=True)
            else:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=False)
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)
        hlpr.task.accumulate_weights(weight_accumulator, local_update)

    hlpr.task.update_global_model(weight_accumulator, global_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = get_current_git_hash()
    if args.debug:
        params['log'] = params['save_model'] = params['wandb'] = False
        params['device'] = torch.device('cpu')

    helper = Helper(params)
    # clean_model = helper.task.train_model_for_sampling()
    if helper.task.clean_model is not None:
        test(helper, helper.task.clean_model)
        test(helper, helper.task.model)

    logger.warning(create_table(params))

    try:
        if helper.params.fl:
            fl_run(helper)
        else:
            run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'runs/{helper.params.name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {helper.params.name}")
        else:
            logger.error(f"Aborted training. No output generated.")
