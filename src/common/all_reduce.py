import torch.distributed as dist
from .dist import DistTrainer
from contextlib import contextmanager


class AllReduceTrainer(DistTrainer):
    def __init__(self, *args, **kwargs):
        super(AllReduceTrainer, self).__init__(*args, **kwargs)
        # This hooks into `Trainer.train` which is a legacy technique
        # drawing inspiration from tnt/engine, perhaps there's a better way
        # Also, on-forward in this case is a misnomer and is in fact called
        # after backward but before update.
        self.on_forward_fn = self.all_reduce_sgd

    @contextmanager
    def on_cpu_for_comm(self):
        # For all-reduce,
        # 1. If we're on cpu, this is unnecessary anyway
        # 2. If we're on gpu, we'll use gloo which doesn't
        #    require transfer to cpu
        assert (not self.args.gpu) or (self.args.comm_backend == 'gloo')
        yield

    def all_reduce_sgd(self):
        """
            All-reduce SGD update
        """
        self.logger.debug('Executing All-reduce SGD update')
        dist.barrier()
        size = float(dist.get_world_size())
        with self.on_cpu_for_comm():
            for param in self.model.parameters():
                dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
                param.grad.data /= size
        dist.barrier()
        self.logger.debug('Done All-reduce SGD update')
        return
