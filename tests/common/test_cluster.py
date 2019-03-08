import torch
from pytest import fixture
from src.common.cluster import Cluster


class TestCluster(object):

    @staticmethod
    @fixture(scope='class', params=[2, 5, 17])
    def mock_args(request, mock_args):
        original = mock_args.num_workers
        mock_args.num_workers = request.param
        yield mock_args
        mock_args.num_workers = original

    @staticmethod
    def identity(x):
        # needs to be pickle-able
        return x

    @staticmethod
    def get_rng_state(_):
        return tuple(torch.get_rng_state().tolist())

    def test_run_processes(self, mock_args):
        cluster = Cluster(mock_args)
        results = cluster.run_processes(self.identity)
        assert results == list(range(mock_args.num_workers))
        return

    def test_seed_non_symmetry(self, mock_args):
        cluster = Cluster(mock_args)
        results = cluster.run_processes(self.get_rng_state)
        assert len(set(results)) == mock_args.num_workers

    def test_consistent_seed_application(self, mock_args):
        cluster_1 = Cluster(mock_args)
        results_1 = cluster_1.run_processes(self.get_rng_state)
        cluster_2 = Cluster(mock_args)
        results_2 = cluster_2.run_processes(self.get_rng_state)
        assert results_1 == results_2
