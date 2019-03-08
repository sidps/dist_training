import logging
import numpy as np
import torch
import torch.distributed as dist
from functools import partial
from hypothesis import given
from hypothesis import strategies as st
from math import floor
from pytest import mark
from src.common.cluster import Cluster
from src.common.pairwise import PairwiseCommTrainer


@given(
    size=st.integers(min_value=2),
    rank_frac=st.floats(min_value=0, max_value=1)
)
def test_peer_selection(size, rank_frac):
    rank = floor(size * rank_frac) % size
    assert rank < size
    peer = PairwiseCommTrainer.select_random(size, rank)
    assert isinstance(peer, int)
    assert peer != rank
    assert rank < size


class TestPairwise(object):
    @staticmethod
    def param_init_check(rank, data, architecture, args):
        logger = logging.getLogger('test_logger.rank_{}'.format(rank))
        trainer = PairwiseCommTrainer(
            rank=rank, data=data, architecture=architecture,
            args=args, logger=logger
        )
        for param in trainer.model.parameters():
            gather_list = [
                torch.zeros(param.data.shape)
                for _ in range(args.num_workers)
            ]
            dist.all_gather(gather_list, param.data)
            dist.barrier()
            for i in range(len(gather_list) - 1):
                assert gather_list[i].tolist() \
                       == gather_list[i + 1].tolist()
        return

    @staticmethod
    def init_comm_check(rank, data, architecture, args):
        logger = logging.getLogger('test_logger.rank_{}'.format(rank))
        trainer = PairwiseCommTrainer(
            rank=rank, data=data, architecture=architecture,
            args=args, logger=logger
        )
        trainer.init_communication()
        assert trainer.peer is not None
        selections = [torch.zeros(1) for _ in range(args.num_workers)]
        dist.all_gather(selections, torch.Tensor([trainer.peer]))
        assert selections[rank][0] == trainer.peer
        requesters = {peer for peer, selection in
                      enumerate(selections) if selection[0] == rank}
        assert requesters == trainer.requesters
        return

    @staticmethod
    def post_comm_check(rank, data, architecture, args):
        logger = logging.getLogger('test_logger.rank_{}'.format(rank))
        trainer = PairwiseCommTrainer(
            rank=rank, data=data, architecture=architecture,
            args=args, logger=logger
        )
        trainer.comm_updates = [
            torch.ones(param.data.shape)
            for param in trainer.model.parameters()
        ]
        expected = [param.data - 1 for param in trainer.model.parameters()]
        trainer.post_comm()
        for exp, param in zip(expected, trainer.model.parameters()):
            assert np.allclose(exp.numpy(), param.data.numpy())
        return

    @mark.parametrize('test_fn', [
        'param_init_check', 'init_comm_check', 'post_comm_check'
    ])
    def test_pairwise(self, mock_args, sample_data,
                      sample_architecture, test_fn):
        # This is required because of how a cluster is run, else
        # each test could have been "independent".
        cluster = Cluster(mock_args)
        # Because for some weird reason involving partial,
        # I can't just pass the method:
        test_fn = getattr(self, test_fn)
        test_fn = partial(
            test_fn, data=sample_data,
            architecture=sample_architecture, args=mock_args,
        )
        cluster.run_processes(test_fn)
        return
