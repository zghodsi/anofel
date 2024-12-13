import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision import datasets, transforms

from model import SimpleRNN, SimpleLinear, SimpleCNN, Lenet5, OldLenet5
from resnet import resnet20
from squeezenet import SqueezeNet
from config import config, Batch
from train import Trainer

from datautils import get_dataset


@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.2f s]' % (time.perf_counter() - time0))


def configure_dataloaders(data_dir: Path) -> Tuple[DataLoader, DataLoader]:
    if config.dataset == 'mnist':
        def create_loader(is_train_loader):
            return DataLoader(
                MNIST(
                    data_dir,
                    train=is_train_loader,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
                ),
                # yield batches for every client
                batch_size=config.n_parties * config.batch_size,
            )
    elif config.dataset == 'cifar10':
        def create_loader(is_train_loader):
            return DataLoader(
                CIFAR10(
                    data_dir,
                    train=is_train_loader,
                    download=True,
                    transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                ),
                # yield batches for every client
                batch_size=config.n_parties * config.batch_size,
            )
    elif config.dataset == 'tiny':
        def create_loader(is_train_loader):
            return DataLoader(
                datasets.ImageFolder(
                    root='data/tiny-imagenet-200/train',
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.4802, 0.4481, 0.3975],
                                             [0.2302, 0.2265, 0.2262])
                    ])
                ),
                # yield batches for every client
                batch_size=config.n_parties * config.batch_size,
            )

    else:
        print("Dataset not supported")
        exit()
    
    return (create_loader(True), create_loader(False))


def configure_model() -> torch.nn.Module:
    #model = SimpleLinear(in_size=28 * 28, out_size=10)
    if config.dataset == 'mnist':
        model = Lenet5()
    elif config.dataset == 'cifar10':
        #model = SimpleCNN(in_size=32 * 32, out_size=10)
        model = resnet20()
    elif config.dataset == 'tiny':
        model = SqueezeNet()

    else:
        print("Dataset not supported")
        exit()
 
    return model


if __name__ == '__main__':
    data_dir = Path(__file__).parent / 'data/'
    data_dir.mkdir(parents=True, exist_ok=True)

    model = configure_model()
    print("num of params: ",  sum(p.numel() for p in model.parameters()))

    #loaders = configure_dataloaders(data_dir)
    #trainer = Trainer(
    #    model=model,
    #    train_loader=loaders[0],
    #    valid_loader=loaders[1],
    #)

    train_dataset, test_dataset, user_groups = get_dataset()
    trainloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    trainer = Trainer(
        model=model,
        train_loader=trainloader,
        valid_loader=testloader,
        train_dataset=train_dataset,
        valid_dataset=test_dataset,
        user_groups=user_groups,
    )


    try:
        trainer.fit()
    except KeyboardInterrupt:
        exit(0)


