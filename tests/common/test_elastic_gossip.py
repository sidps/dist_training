import logging
import numpy as np
import torch
import torch.distributed as dist
from functools import partial
from pytest import mark
from src.common.cluster import Cluster
from src.common.elastic_gossip import ElasticGossipTrainer


class TestElasticGossip(object):
    @staticmethod
    def elastic_gossip_check(rank, data, architecture, args):
        logger = logging.getLogger('test_logger.rank_{}'.format(rank))
        trainer = ElasticGossipTrainer(
            rank=rank, data=data, architecture=architecture,
            args=args, logger=logger
        )
        # re-init - hack to make parameters asymmetric
        trainer.model = architecture.get_new_model()

        # fix peers
        trainer.peer = (rank + 1) % args.num_workers
        trainer.requesters = [(rank - 1) % args.num_workers]

        trainer.compute_comm_updates()

        # collect peer's params
        peer_params_list = [
            [
                torch.zeros(param.data.shape)
                for _ in range(args.num_workers)
            ]
            for param in trainer.model.parameters()
        ]
        for peer_params, param, update in \
                zip(peer_params_list, trainer.model.parameters(),
                    trainer.comm_updates):
            dist.all_gather(peer_params, param.data)
            dist.barrier()
            # ensure params are unequal
            assert len(set(repr(r.tolist()) for r in peer_params)) \
                == args.num_workers
            # ensure updates are as expected
            exp = args.elastic_alpha * \
                sum([(param.data - peer_params[x])
                     for x in [trainer.peer, trainer.requesters[0]]])
            assert np.allclose(exp, update)

        return

    @mark.parametrize('test_fn', [
        'elastic_gossip_check',
    ])
    def test_elastic_gossip(self, mock_args, sample_data,
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
