import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import transforms
import dataset.mnist as mnist_dataset
from models.simple import SimpleNet
from tasks.task import Task
import torch


class MNISTTask(Task):
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    def load_data(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # self.normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # self.normalize
        ])

        self.train_dataset = mnist_dataset.FashionMNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)
        if self.params.compute_grads_only:
            sampler = torch_data.SequentialSampler(self.train_dataset)
        elif self.params.recover_indices:
            indices_results = torch.load(self.params.recover_indices)
            if self.params.poisoning_proportion == 0.0:
                weights = torch.ones_like(indices_results['weights'])
                weights[indices_results['indices'].nonzero()] = 0.0
            else:
                weights = torch.clamp(indices_results['weights'], min=0.01)
                weights = 1/weights
                if self.params.cut_grad_threshold:
                    weights[indices_results['weights'] > self.params.cut_grad_threshold] = 0.0
                    weights[indices_results[
                                'weights'] <= self.params.cut_grad_threshold] = 1.0
                    print(f'Shape: {weights.shape}, sum: {weights.sum()}')
            weights = weights/weights.sum()

            sampler = torch_data.WeightedRandomSampler(weights, len(self.train_dataset))
        else:
            sampler = torch_data.RandomSampler(self.train_dataset)
        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=False,
                                                  sampler=sampler,
                                                  num_workers=0)
        self.test_dataset = mnist_dataset.FashionMNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=0)

        self.test_attack_dataset = mnist_dataset.FashionMNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)

        self.test_attack_loader = torch_data.DataLoader(self.test_attack_dataset,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=0)
        self.classes = self.train_dataset.classes
        return True

    def build_model(self):
        return SimpleNet(num_classes=len(self.classes))
