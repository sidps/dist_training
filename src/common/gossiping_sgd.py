import torch
import torch.distributed as dist
from .pairwise import PairwiseCommTrainer


class GossipingSGDPullTrainer(PairwiseCommTrainer):
    """ Gossiping SGD - pull variant. """

    def __init__(self, *args, **kwargs):
        super(GossipingSGDPullTrainer, self).__init__(*args, **kwargs)

    def compute_comm_updates(self):
        if (self.peer is None) and (not self.requesters):
            return

        self.logger.debug('Computing gossiping sgd (pull) updates')
        self.comm_updates = []
        with self.on_cpu_for_comm():
            for param in self.model.parameters():
                self.logger.debug('Sending and receiving param(s)')
                # A container to hold async requests and param sets
                requests = []
                buffer = torch.zeros_like(param.data)

                if self.peer is not None:
                    self.logger.debug('Initiating irecv request with own '
                                      'peer: rank %s' % self.peer)
                    requests.append(dist.irecv(
                        tensor=buffer,
                        src=self.peer
                    ))

                for peer in self.requesters:
                    self.logger.debug('Initiating isend request with '
                                      'requesting peer: rank %s' % peer)
                    requests.append(dist.isend(
                        tensor=param.data,
                        dst=peer
                    ))

                # Wait for all the requests to complete
                for r in requests:
                    r.wait()
                self.logger.debug('Requests complete')

                if self.peer is None:
                    continue

                # Then compute the Gossiping SGD update.
                s = param.data - buffer
                s /= 2
                self.comm_updates.append(s)
                self.logger.debug('Finished computing average '
                                  'for parameter set')

        self.logger.debug('Done computing gossiping sgd (pull) updates')
        return
