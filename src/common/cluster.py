import os
import torch
import torch.distributed as dist
from torch import multiprocessing
from functools import partial


class Cluster(object):
    def __init__(self, args):
        super(Cluster, self).__init__()
        self.args = args

    def _init_worker(self, rank, fn):
        """
            Run a single process in a torch.distributed environment

            :param rank: Rank of a given worker
            :type rank: int

            :param fn: Function to run on worker (passes rank as arg)
            :type fn: function

            :return: result of running fn
        """
        os.environ['MASTER_ADDR'] = self.args.master_addr
        os.environ['MASTER_PORT'] = self.args.master_port

        # Below is an attempt to address issue with number of threads:
        # (https://github.com/pytorch/pytorch/issues/975)
        os.environ['OMP_NUM_THREADS'] = str(self.args.num_threads)
        torch.set_num_threads(self.args.num_threads)

        # break symmetry in random seed across workers
        seed = self.args.seed + rank
        torch.manual_seed(seed)
        if self.args.gpu:
            torch.manual_seed(seed)

        dist.init_process_group(self.args.comm_backend, rank=rank,
                                world_size=self.args.num_workers)
        res = fn(rank)
        return res

    def run_processes(self, fn):
        """ Spawn a set of processes using multi-processing.

            :param fn: a function to run
            :type fn: function

            :returns: results of the function mapped to rank of each worker
            :rtype: list
        """
        if self.args.gpu:
            Pool = multiprocessing.get_context('spawn').Pool
        else:
            Pool = multiprocessing.Pool

        partial_init = partial(self._init_worker, fn=fn)
        pool = Pool(
            self.args.num_workers,
            maxtasksperchild=1,
        )
        results = pool.map(
            partial_init,
            range(self.args.num_workers),
            chunksize=1,
        )
        pool.close()
        return results

    # This is used to run a single worker in a distributed system
    # This is considered a stub can add more functionality later
    run_single = _init_worker
