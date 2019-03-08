import os
import logging
import bunyan
import torch
import numpy as np
from sklearn.metrics import pairwise_distances


def setup_logger(log_path, log_level, rank=None, append=False):
    if rank is None:
        log_file = 'main.log'
        logger_id = 'Main'
    else:
        log_file = 'rank_%s.log' % rank
        logger_id = 'Rank-%s' % rank
    log_file = os.path.join(log_path, log_file)
    handler = logging.FileHandler(
        log_file,
        mode='a' if append else 'w',
    )
    handler.setFormatter(bunyan.BunyanFormatter())
    logger = logging.getLogger(logger_id)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def teardown_logger(logger):
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    return


def get_checkpoint_path(log_path, rank):
    return os.path.join(log_path, 'rank_%s.cp.pt' % rank)


def save_checkpoint(log_path, rank, state):
    checkpoint_path = get_checkpoint_path(log_path, rank)
    torch.save(state, checkpoint_path)
    return


def load_checkpoint(log_path, rank):
    checkpoint_path = get_checkpoint_path(log_path, rank)
    state = torch.load(
        checkpoint_path,
        map_location=lambda storage, location: storage
    )
    return state


def generate_topo_based_weights(num_workers, penalty=1, eps=1e-5):
    if penalty < 0:
        raise ValueError('penalty must be >= 0, got %f' % penalty)
    if eps <= 0:
        raise ValueError('eps must be > 0, got %f' % eps)
    if num_workers < 2:
        raise NotImplementedError('No support for num_workers < 2,'
                                  ' got %d' % num_workers)

    # Generate random "locations" on 1x1 square
    locs = torch.rand(num_workers, 2)

    # Compute distances between locations
    dists = pairwise_distances(locs.numpy(), metric='euclidean')

    # The above is a pairwise-distance matrix (n x n).
    # 1. Shift each row such that the first element corresponds to self
    #    i.e. shift the diagonal to be the first column
    # 2. Then drop the first column because we don't care about distance
    #    to self
    for i in range(len(dists)):
        dists[i, :] = np.concatenate((
            dists[i, i:], dists[i, :i]
        ))
    dists = dists[:, 1:]

    # Add eps so distance is never 0
    dists += eps

    # Compute weights as 1 / (distance ^ factor)
    # factor == 0 => even weights
    # higher the factor, more the distance is penalized
    weights = 1 / (dists ** penalty)

    # Normalize so each row sums to 1
    weight_sum = weights.sum(axis=1)
    weight_sum = np.broadcast_to(
        weight_sum,
        weights.transpose().shape
    ).transpose()
    weights /= weight_sum

    return locs.tolist(), weights.tolist()
