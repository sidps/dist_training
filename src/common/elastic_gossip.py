import torch
import torch.distributed as dist
from .pairwise import PairwiseCommTrainer


class ElasticGossipTrainer(PairwiseCommTrainer):
    def __init__(self, *args, **kwargs):
        super(ElasticGossipTrainer, self).__init__(*args, **kwargs)

    def compute_comm_updates(self):
        # In elastic gossip, it doesn't matter who initiated,
        # both peers exchange params
        peers = set(self.requesters)
        if self.peer is not None:
            peers.add(self.peer)

        if not peers:
            return

        self.logger.debug('Computing elastic gossip updates')
        self.comm_updates = []
        with self.on_cpu_for_comm():
            for param in self.model.parameters():
                self.logger.debug('Sending and receiving param(s)')
                # Containers to hold async requests and param sets
                requests = []
                buffers = [
                    torch.zeros_like(param.data)
                    for _ in peers
                ]

                for peer, buffer in zip(peers, buffers):
                    self.logger.debug('Initiating requests with peer: '
                                      'rank %s' % peer)
                    requests.append(dist.isend(
                        tensor=param.data,
                        dst=peer
                    ))
                    requests.append(dist.irecv(
                        tensor=buffer,
                        src=peer
                    ))

                # Wait for all the requests to complete
                for r in requests:
                    r.wait()
                self.logger.debug('Requests complete for param set')

                # Then compute the elastic update and stash
                s = len(buffers) * param.data
                s -= sum(buffers)
                s *= self.args.elastic_alpha
                self.comm_updates.append(s)
                self.logger.debug('Update computed for param set')

        self.logger.debug('Done computing elastic gossip updates')
        return
