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
        dataset_opts = dict(
            root=self.args.data['data_path'],
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    self.args.data['mean'],
                    self.args.data['std'],
                )
            ])
        )
        self.train_dataset = datasets.MNIST(
            train=True,
            download=True,
            **dataset_opts
        )
        train_val_indexes = randperm(len(self.train_dataset))
        val_size = self.args.data['val_size']
        self.val_indexes = train_val_indexes[:val_size]
        self.train_indexes = train_val_indexes[val_size:]
        self.test_dataset = datasets.MNIST(
            train=False,
            **dataset_opts
        )
        return
