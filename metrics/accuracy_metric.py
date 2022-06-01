import torch
from metrics.metric import Metric


class AccuracyMetric(Metric):

    def __init__(self, top_k=(1,), drop_label=None, total_dropped=None):
        self.name = 'Accuracy'
        self.top_k = top_k
        self.main_metric_name = 'Top-1'
        self.preds = list()
        self.ground_truth = list()
        self.drop_label = drop_label
        self.total_dropped = total_dropped
        super().__init__(name='Accuracy', train=False)

    def compute_metric(self, outputs: torch.Tensor,
                       labels: torch.Tensor):
        """Computes the precision@k for the specified values of k"""
        max_k = max(self.top_k)
        batch_size = labels.shape[0]

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        self.preds.append(pred.detach().cpu())
        self.ground_truth.append(labels.detach().cpu())

        res = dict()
        for k in self.top_k:
            correct_k = correct[:k].view(-1).float().sum(0)
            res[f'Top-{k}'] = (correct_k.mul_(100.0 / batch_size)).item()

        if self.drop_label:
            drop_label_pos = (labels.view(1, -1).expand_as(pred)[0] == self.drop_label).nonzero().view(-1)
            correct_drop = pred[0][drop_label_pos].eq(self.drop_label).sum().cpu()
            res[f'Drop_{self.drop_label}'] = 100 * correct_drop.item()/self.total_dropped

        return res
