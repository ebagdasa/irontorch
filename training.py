import argparse
import shutil
import torch
from datetime import datetime

import yaml
from prompt_toolkit import prompt
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from utils.utils import *

logger = logging.getLogger('logger')


def get_percentage(params, train_dataset, batch):
    attack_count = 0
    drop_label = 0
    for i, x in enumerate(batch.indices):
        if train_dataset.true_targets[x].item() != params.backdoor_label and \
              batch.aux[i].item() == 1:
            attack_count += 1
        if params.drop_label is not None and train_dataset.targets[x] == params.drop_label:
            drop_label += 1
    return 100.0 * attack_count/batch.indices.shape[0],  100.0 * drop_label/batch.indices.shape[0]


def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True):
    criterion = hlpr.task.criterion
    model.train()

    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)
        optimizer.zero_grad(set_to_none=True)
        if hlpr.params.label_noise:
            size = int(hlpr.params.label_noise * batch.labels.shape[0])
            batch.labels[:size] = torch.randint(0, 10, [size,], device=batch.labels.device)
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)
        attack_percent, drop_label = get_percentage(hlpr.params, hlpr.task.train_dataset, batch)
        hlpr.params.running_losses['attack_percent'].append(attack_percent)
        hlpr.params.running_losses['drop_label'].append(drop_label)
        if hlpr.params.saved_grads and hlpr.params.opacus:
            optimizer.batch_idx = i
            optimizer.data_accum[i] = batch.indices.detach().cpu()
            optimizer.label_accum[i] = batch.labels.detach().cpu()
            optimizer.aux[i] = batch.aux.detach().cpu()
            optimizer.loss_accum[i] = loss.detach().cpu()

        loss.backward()
        if hlpr.params.batch_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hlpr.params.grad_clip)
            for param in model.parameters():
                noised_layer = torch.FloatTensor(param.shape)
                noised_layer = noised_layer.to(param.device)
                noised_layer.normal_(mean=0, std=hlpr.params.grad_sigma)
                param.grad.add_(noised_layer)
        optimizer.step()

        hlpr.report_training_losses_scales(i, epoch)
        if i == hlpr.params.max_batch_id:
            break
    hlpr.task.scheduler_step()
    return


def test(hlpr: Helper, model, backdoor=False):
    model.eval()
    hlpr.task.reset_metrics()
    test_loader = hlpr.task.test_attack_loader if backdoor else hlpr.task.test_loader
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            batch = hlpr.task.get_batch(i, data)
            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    prefix = 'Backdoor' if backdoor else 'Normal'
    hlpr.report_metrics(prefix=f'Test/{prefix}')
    # metric = hlpr.task.report_metrics(epoch,
    #                          prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
    #                          tb_writer=hlpr.tb_writer,
    #                          tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return None


def run(hlpr):
    # acc = test(hlpr, 0, backdoor=False)
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        acc = test(hlpr, hlpr.task.model, backdoor=False)
        hlpr.plot_confusion_matrix(backdoor=False, epoch=epoch)
        test(hlpr, hlpr.task.model, backdoor=True)
        hlpr.plot_confusion_matrix(backdoor=True, epoch=epoch)
        hlpr.save_model(hlpr.task.model, epoch, acc)


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
