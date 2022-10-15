import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.resnet import resnet18
from tasks.task import Task
from dataset.imagenet import ImageNet


class ImagenetTask(Task):

    def load_data(self):

        train_transform = transforms.Compose([
            transforms.Resize(132),
            transforms.RandomResizedCrop(128),
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        test_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(132),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            self.normalize,
            transforms.RandomErasing(p=self.params.transform_erase,
                                                                scale=(0.01, 0.09),
                                                                ratio=(0.3, 3.3), value=2,
                                                                inplace=False)
        ])

        self.train_dataset = ImageNet(
            root=self.params.data_path,
            split='train', transform=train_transform)

        self.test_dataset = ImageNet(
            root=self.params.data_path,
            split='val', transform=test_transform)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=2)

        with open(
                f'{self.params.data_path}/imagenet1000_clsidx_to_labels.txt') \
                as f:
            self.classes = eval(f.read())

    def make_loaders(self):
        from ffcv.pipeline.operation import Operation
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
            RandomHorizontalFlip, ToTorchImage
        from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
            RandomResizedCropRGBImageDecoder
        from ffcv.fields.basics import IntDecoder
        from pathlib import Path
        import numpy as np
        import torch

        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
        IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
        DEFAULT_CROP_RATIO = 224 / 256

        def get_resolution(epoch, min_res=160,
                           max_res=192, end_ramp=13, start_ramp=11):
            assert min_res <= max_res

            if epoch <= start_ramp:
                return min_res

            if epoch >= end_ramp:
                return max_res

            # otherwise, linearly interpolate to the nearest multiple of 32
            interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
            final_res = int(np.round(interp[0] / 32)) * 32
            return final_res

        this_device = f'cuda:0'

        res = get_resolution(epoch=0)
        decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device), non_blocking=True)
        ]

        order = OrderOption.QUASI_RANDOM
        self.train_loader = Loader('/home/eugene/data/ffcv/imagenet/train_200_0.50_70.ffcv',
                        batch_size=self.params.batch_size,
                        num_workers=10,
                        order=order,
                        os_cache=True,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=False)

        res_tuple = (256, 256)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]
        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device),
                     non_blocking=True)
        ]


        self.test_loader = Loader('/home/eugene/data/ffcv/imagenet/val_200_0.50_70.ffcv',
                        batch_size=self.params.batch_size,
                        num_workers=10,
                        order=order,
                        os_cache=True,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=False)

        self.val_loader = self.test_loader
        self.val_attack_loaders['Primitive'] = self.test_loader


    def build_model(self) -> None:
        return resnet18(pretrained=self.params.pretrained)
