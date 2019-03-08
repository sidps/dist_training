import torch.distributed as dist
from .dist import DistTrainer
from contextlib import contextmanager


class NoCommTrainer(DistTrainer):
    def __init__(self, *args, **kwargs):
        super(NoCommTrainer, self).__init__(*args, **kwargs)
        # This hooks into `Trainer.train` which is a legacy technique
        # drawing inspiration from tnt/engine, perhaps there's a better way
        self.on_forward_fn = self.no_comm

    @contextmanager
    def on_cpu_for_comm(self):
        # For all-reduce,
        # 1. If we're on cpu, this is unnecessary anyway
        # 2. If we're on gpu, we'll use gloo which doesn't
        #    require transfer to cpu
        assert (not self.args.gpu) or (self.args.comm_backend == 'gloo')
        yield

    def no_comm(self):
        self.logger.debug('No comm - simply syncing')
        dist.barrier()
        return
