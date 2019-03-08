from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import randperm


class BaseData(object):
    """ Base class for data classes used in each experiment. """

    def __init__(self, args):
        """
            Needs either val-dataset or combination of train and val
            indexes, in which case the train and val instances are both
            sampled from the train-dataset.

            The latter is useful for torchvision datasets that don't
            have a splitting mechanism otherwise.
        """
        self.args = args
        self.train_dataset = None
        self.val_dataset = None
        self.train_indexes = None
        self.val_indexes = None
        self.test_dataset = None
        self.splits = None

    def get_train(self, split_idx=None):
        # To keep the aggregate batch size constant irrespective of the
        # number of workers, the batch size for each worker is the
        # given batch size (assumed to be aggregate) divided by the
        # number of workers.
        if self.args.batch_size % self.args.num_workers != 0:
            raise ValueError('(aggregate) batch-size must be a '
                             'multiple of number of workers')
        worker_batch_size = self.args.batch_size / self.args.num_workers
        opts = dict(
            dataset=self.train_dataset,
            batch_size=worker_batch_size,
        )
        if split_idx is not None:
            opts['sampler'] = \
                SubsetRandomSampler(self.splits[split_idx])
        else:
            opts['shuffle'] = True
        if self.args.gpu:
            opts['pin_memory'] = True
        return DataLoader(**opts)

    def get_val(self):
        assert self.val_dataset is not None or \
            (self.val_indexes is not None and
             self.train_indexes is not None),\
            'need either val-dataset or indexes to select from train'
        opts = {}
        if self.args.gpu:
            opts['pin_memory'] = True
        if self.val_dataset:
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=self.args.val_batch_size or
                           len(self.val_dataset),
                **opts,
            )
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.val_batch_size or len(self.val_indexes),
            sampler=SubsetRandomSampler(self.val_indexes),
            **opts
        )

    def get_test(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.args.test_batch_size or len(self.test_dataset),
        )

    def make_splits(self):
        if self.train_indexes is not None:
            assert self.val_indexes is not None and \
                self.val_dataset is None, 'something very wrong'
            indexes = self.train_indexes
        else:
            l = len(self.train_dataset)
            indexes = randperm(l)
        # FIXME: fails when l % num_splits != 0
        if len(indexes) % self.args.num_workers != 0:
            raise ValueError('Num-workers should divide length of '
                             'training set.')
        self.splits = indexes.view(self.args.num_workers, -1)
        return
