import pickle
from sklearn.datasets import load_iris
from pathlib import Path
from torch.utils.data import TensorDataset
from torch import FloatTensor, LongTensor, randperm
from ..common.base_data import BaseData


class Data(BaseData):
    def __init__(self, args):
        self.args = args
        super(Data, self).__init__(self.args)
        self._setup()
        self.make_splits()

    def _get_indexes(self):
        indexes_path = Path(self.args.data['indexes_path'])
        try:
            with indexes_path.open('rb') as f:
                indexes = pickle.load(f)
        except:
            if not indexes_path.parent.exists():
                indexes_path.parent.mkdir(parents=True, exist_ok=True)

            indexes = randperm(150)
            with indexes_path.open('wb') as f:
                pickle.dump(indexes, f)

        return indexes

    def _setup(self):
        iris = load_iris()
        data = FloatTensor(iris.data)
        target = LongTensor(iris.target)

        indexes = self._get_indexes()
        train_indexes = indexes[:100]
        val_indexes = indexes[100:125]
        test_indexes = indexes[125:]

        self.train_dataset = TensorDataset(
            data[train_indexes],
            target[train_indexes]
        )
        self.val_dataset = TensorDataset(
            data[val_indexes],
            target[val_indexes]
        )
        self.test_dataset = TensorDataset(
            data[test_indexes],
            target[test_indexes]
        )
        return
