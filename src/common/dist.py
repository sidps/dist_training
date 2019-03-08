import torch.distributed as dist
from contextlib import contextmanager
from .train import Trainer


class DistTrainer(Trainer):
    """ Base class for all distributed trainers. """

    def __init__(self, *args, **kwargs):
        super(DistTrainer, self).__init__(*args, **kwargs)
        if not self.args.from_checkpoint:
            self.init_parameters()

    @contextmanager
    def on_cpu_for_comm(self):
        # Communication often entails cpu <=> gpu transfer,
        # which this method should handle.
        # Subclasses must override
        raise NotImplementedError

    def init_parameters(self):
        """
            Initialize the parameters for each worker with the
            parameters used by the rank-0 worker.

            The alternative could be to use the same random number
            generator initialized with the same seed across all workers.
        """
        dist.barrier()
        with self.on_cpu_for_comm():
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
        dist.barrier()
        return

    def agg_parameters(self):
        """
            All-reduce the parameters from each worker and average.
            Useful for post training evaluation.
        """
        dist.barrier()
        size = float(dist.get_world_size())
        with self.on_cpu_for_comm():
            for param in self.model.parameters():
                dist.all_reduce(param.data, op=dist.reduce_op.SUM)
                param.data /= size
        dist.barrier()
        self.logger.debug('Done agg parameters')
        return

    def train(self):
        super().train()
        self.agg_parameters()
        if self.rank != 0:
            return
        acc = self.evaluate()
        self.logger.info({
            'Rank': self.rank,
            'AggTestAcc': acc,
        })
        self.logger.info({
            'Rank': self.rank,
            'AggTestAccPerc': '%.2f %%' % (acc * 100),
        })
        return
