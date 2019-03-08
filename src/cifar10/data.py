from torchvision import datasets, transforms
from torch import randperm
from ..common.base_data import BaseData


class Data(BaseData):
    def __init__(self, args):
        self.args = args
        super(Data, self).__init__(self.args)
        self._setup()
        self.make_splits()

    def _setup(self):
        testset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                self.args.data['mean'],
                self.args.data['std'],
            )
        ])
        trainset_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            testset_transform
        ])
        self.train_dataset = datasets.CIFAR10(
            train=True,
            download=True,
            root=self.args.data['data_path'],
            transform=trainset_transform,
        )
        train_val_indexes = randperm(len(self.train_dataset))
        val_size = self.args.data['val_size']
        self.val_indexes = train_val_indexes[:val_size]
        self.train_indexes = train_val_indexes[val_size:]
        self.test_dataset = datasets.CIFAR10(
            train=False,
            root=self.args.data['data_path'],
            transform=testset_transform,
        )
        return
