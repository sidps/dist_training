import torch
import torch.distributed as dist
from contextlib import ExitStack, contextmanager
from .dist import DistTrainer


class PairwiseCommTrainer(DistTrainer):
    def __init__(self, *args, **kwargs):
        self.peer = None
        self.requesters = []
        self.comm_updates = []
        super(PairwiseCommTrainer, self).__init__(*args, **kwargs)

        # Convert weights to intervals if we're using weighted peer
        # selection in gossip-like protocols
        self.weight_intervals = None
        self.setup_weight_intervals()

        # These hook into `Trainer.train` which is a legacy technique
        # drawing inspiration from torchnet/engine, perhaps there's a
        # better way
        self.on_forward_fn = self.communicate
        self.on_update_fn = self.post_comm

    def setup_weight_intervals(self):
        if not self.args.weighted_gossip:
            self.weight_intervals = None
            return
        cum_sum = 0
        self_weights = self.args.gossip_weights[self.rank]
        self.weight_intervals = [None for _ in
                                 range(len(self_weights) + 1)]
        for i, w in enumerate(self_weights):
            self.weight_intervals[i] = (cum_sum, cum_sum + w)
            cum_sum += w
        self.weight_intervals[-1] = (cum_sum, 1)
        return

    @contextmanager
    def on_cpu_for_comm(self):
        assert (not self.args.gpu) or (self.args.comm_backend == 'tcp')
        with ExitStack() as stack:
            if self.args.gpu:
                stack.enter_context(torch.cuda.device(self.device_id))
                self.model.cpu()
            self.logger.debug('On CPU for comm')
            yield
            if self.args.gpu:
                self.model.cuda()
                for i in range(len(self.comm_updates)):
                    self.comm_updates[i] = \
                        self.comm_updates[i].cuda(
                            self.device_id
                        )
            self.logger.debug('Off CPU after comm')

    @staticmethod
    def dist_all_gather(val):
        selections = [
            torch.zeros(1)
            for _ in range(dist.get_world_size())
        ]
        gather_val = torch.Tensor([val]) if val is not None \
            else torch.Tensor([float('nan')])

        dist.barrier()
        dist.all_gather(selections, gather_val)
        dist.barrier()

        selections = map(lambda x: x[0], selections)
        return selections

    @staticmethod
    def bernoulli_trial(prob):
        trial = torch.bernoulli(torch.Tensor([prob]))
        trial = bool(trial[0])
        return trial

    @staticmethod
    def select_random(size, rank, weight_intervals=None):
        """ Select a random peer in the cluster."""
        if weight_intervals is None:
            # Select a peer uniformly at random
            # 1. int between `0` and `size` exclusive
            # 2. right-shift by `rank`
            rand_int = (torch.floor(torch.rand(1) * (size - 1)))[0]
            peer = (int(rand_int) + 1 + rank) % size
            return peer

        # Split the interval [0, 1) into intervals whose widths
        # correspond to the weights (this is already done with
        # `weight_intervals`). Then
        random_val = torch.rand(1)[0]
        for i, w in enumerate(weight_intervals):
            if w[0] <= random_val < w[1]:
                return (i + 1 + rank) % size
        assert False, 'unsure why we got here, re-evaluate life'

    def init_communication(self):
        """
            Determines peers for the round of gossip.
        """
        self.peer = None
        self.requesters = None

        # Check if we're at agg_period
        if self.update % self.args.agg_period != 0:
            return

        # Check through coin toss if self will initiate
        if self.bernoulli_trial(self.args.agg_prob):
            # Select a random peer.
            self.peer = self.select_random(
                self.args.num_workers,
                self.rank,
                self.weight_intervals
            )
        self.logger.debug('Will communicate with: %s' % self.peer)

        # dist.all_gather these selections so all workers know
        # every other worker's selection
        selections = self.dist_all_gather(self.peer)
        self.logger.debug({'selections': selections})

        # Collect the set of workers that will comm with self.
        self.requesters = {peer for peer, selection in
                           enumerate(selections) if selection == self.rank}

        self.logger.debug({
            'own_peer': self.peer,
            'requesting_peers': self.requesters
        })
        return

    def communicate(self):
        self.init_communication()
        self.compute_comm_updates()
        return

    def compute_comm_updates(self):
        # Subclasses should override
        # 1. populate comm_updates with updates for each param-set
        raise NotImplementedError

    def post_comm(self):
        # If we're gossiping, then add the gossip component we got
        # in the pre-update step
        if not self.comm_updates:
            return

        self.logger.debug('Beginning post_comm.')

        # Note that updates are subtracted
        for i, param in enumerate(self.model.parameters()):
            param.data -= self.comm_updates[i]

        self.comm_updates = []
        self.logger.debug('Done post_comm')
        return
