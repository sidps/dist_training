import logging
import numpy as np
import torch
import torch.distributed as dist
from copy import deepcopy
from functools import partial
from pytest import mark
from torch.autograd import Variable
from src.common.all_reduce import AllReduceTrainer
from src.common.cluster import Cluster


class TestAllReduce(object):
    def param_init_check(self, rank, data, architecture, args):
        logger = logging.getLogger('test_logger.rank_{}'.format(rank))
        trainer = AllReduceTrainer(
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

    def all_reduce_check(self, rank, data, architecture, args):
        logger = logging.getLogger('test_logger_rank{}'.format(rank))
        trainer = AllReduceTrainer(
            rank=rank, data=data, architecture=architecture,
            args=args, logger=logger
        )
        # Compute the forward and backward passes
        inputs, target = next(iter(trainer.train_loader))
        inputs, target = Variable(inputs), Variable(target)
        out = trainer.model(inputs)
        loss = trainer.loss_fn(out, target)
        loss.backward()
        before = [
            [
                torch.zeros(param.data.shape)
                for _ in range(args.num_workers)
            ]
            for param in trainer.model.parameters()
        ]
        after = deepcopy(before)
        # collect gradients from each worker and ensure they're unequal
        for before_list, param in zip(before, trainer.model.parameters()):
            dist.all_gather(before_list, param.grad.data)
            dist.barrier()
            # ensure grads are unequal
            assert len(set(repr(b.tolist()) for b in before_list)) \
                == args.num_workers
        # run on-forward hook which should perform all-reduce on gradients
        trainer.on_forward_fn()
        # collect gradients again and ensure they're equal,
        # and are the average of gradients collected earlier
        for after_list, before_list, param \
                in zip(after, before, trainer.model.parameters()):
            dist.all_gather(after_list, param.grad.data)
            dist.barrier()
            # ensure params grads are equal
            assert len(set(repr(a.tolist()) for a in after_list)) == 1
            # ensure params grads are averaged
            exp_avg = np \
                .vstack([b.numpy().ravel() for b in before_list]) \
                .mean(axis=0)
            act_avg = after_list[0].numpy().ravel()
            assert np.allclose(exp_avg, act_avg)
        return

    @mark.parametrize('test_fn', [
        'param_init_check', 'all_reduce_check'
    ])
    def test_all_reduce(self, mock_args, sample_data,
                        sample_architecture, test_fn):
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
