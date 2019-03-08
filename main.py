"""
    This can be run two ways:
    1. `python main.py` will spawn multiple processes in the same node
       and run simulations over them.
    2. This can be alternately run by externally spawning multiple
       instances of this program and using the --rank argument.
       The `rank`s have to be unique for every spawned instance and has
       to be in the range [0, --num-workers).
       While a bit involved, this technique allows more flexibility
       than multiprocessing. For ex.:
       ```
       $ echo {0..3} | xargs -n 1 -I {} -P 4 \
            python main.py --epochs=50 --rank={} --num-workers=4
       ```
"""

__version__ = '0.2.1'


import torch
import logging
import os
import json
from datetime import datetime
from argparse import ArgumentParser
from config import configs
from src.common.util import get_checkpoint_path, \
    generate_topo_based_weights
from src.iris.experiment import Experiment as IrisExperiment
from src.mnist.experiment import Experiment as MnistExperiment
from src.cifar10.experiment import Experiment as Cifar10Experiment
from src.common.train import Trainer as LocalTrainer
from src.common.all_reduce import AllReduceTrainer
from src.common.elastic_gossip import ElasticGossipTrainer
from src.common.gossiping_sgd import GossipingSGDPullTrainer
from src.common.no_comm import NoCommTrainer


EXPERIMENTS = dict(
    iris=IrisExperiment,
    mnist=MnistExperiment,
    cifar10=Cifar10Experiment,
)

TRAINERS = dict(
    local=LocalTrainer,
    noComm=NoCommTrainer,
    gradAllReduce=AllReduceTrainer,
    elasticGossip=ElasticGossipTrainer,
    gossipingSgd=GossipingSGDPullTrainer,
)


def parse_master_loc(args):
    addr, port = args.master_loc.split(':')
    if not addr:
        raise ValueError('Master address must be provided.')
    if not port:
        port = '29500'
        logging.warning('Master port not provided, using %s' % port)
    args.master_addr = addr
    args.master_port = port
    return args


def init_log_path(args):
    if not args.log_path:
        args.log_path = os.path.join('logs/', args.exp_id)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    return args


def checkpoint_check(args):
    if args.from_checkpoint:
        expected_paths = (
            get_checkpoint_path(args.log_path, r)
            for r in range(args.num_workers)
        )
        if not all(os.path.exists(p) for p in expected_paths):
            print('Warning: At least some workers are missing '
                  'checkpoint, starting from scratch.')
            args.from_checkpoint = False
    return args


def set_default_args(args):
    hp = configs[args.experiment]['hyper_parameters']
    args.epochs = args.epochs or hp['num_epoch']
    args.batch_size = args.batch_size or hp['batch_size']
    args.learning_rate = args.learning_rate or hp['learning_rate']
    args.momentum = args.momentum or hp['momentum']
    args.nesterov = args.nesterov or hp['nesterov']
    args.anneal_milestones = args.anneal_milestones \
        or hp['anneal_milestones']
    args.anneal_factor = args.anneal_factor \
        or hp['anneal_factor']
    args.data = configs[args.experiment]['data']
    return args


def gpu_check(args):
    args.gpu = args.gpu and torch.cuda.is_available() and \
        torch.cuda.device_count() > 0
    args.eval_on_gpu = args.eval_on_gpu and args.gpu
    return args


def save_input_args_as_metadata(args):
    if args.rank:
        # write only if in multiprocessing mode or if rank 0
        return
    metadata_path = os.path.join(args.log_path, 'metadata.json')
    d = vars(args)
    d['__version__'] = __version__
    with open(metadata_path, 'w') as f:
        json.dump(d, f)
    return


def trainer_check(args):
    if args.num_workers == 1:
        args.agg_method = 'local'
    return args


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {}'.format(__version__)
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=0,
        help='number of epochs, 0 falls back to default for exp '
             'specified in configs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='learning rate, default based on exp config'
    )
    parser.add_argument(
        '--anneal-milestones',
        type=int,
        nargs='+',
        help='if annealing, when to anneal, default based on exp config'
    )
    parser.add_argument(
        '--anneal-factor',
        type=float,
        help='if annealing, how much to anneal by, default based on exp'
             'config'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        help='momentum, default based on exp config'
    )
    parser.add_argument(
        '--no-nesterov',
        action='store_false',
        dest='nesterov',
        help='use Nesterov momentum'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0,
        help='weight decay'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='number of updates between logs'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=None,
        help='number of epochs between checkpoints'
    )
    parser.add_argument(
        '--from-checkpoint',
        action='store_true',
        help='Start training from last checkpoint.\n'
             'Note: anneal milestones aren\'t saved in state,'
             '      so input needs to be adjusted accordingly.'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='python logging level'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='The aggregate batch-size per training update, '
             'the batch per worker is this value over num workers.'
    )
    parser.add_argument(
        '--val-batch-size',
        type=int,
        help='The batch-size to use for validation set, '
             'this only has consequences on resource consumption.'
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        help='The batch-size to use for test set, '
             'this only has consequences on resource consumption.'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='total number of workers'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=1,
        help='number of threads per worker'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='iris',
        choices=tuple(EXPERIMENTS.keys())
    )
    parser.add_argument(
        '--exp-id',
        type=str,
        default='unspecified',
        help='a unique identifier for this run, to help with logging '
             'and such'
    )
    parser.add_argument(
        '--log-path',
        type=str,
        help='to override the path determined by `exp-id`'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_false',
        dest='gpu',
        help='Use the gpu?'
    )
    parser.add_argument(
        '--no-eval-on-gpu',
        action='store_false',
        dest='eval_on_gpu',
        help='Use the gpu for validation as well?'
    )
    parser.add_argument(
        '--sync-cuda',
        action='store_false',
        dest='async_cuda',
        help='Async transfer of data to gpu'
    )
    parser.add_argument(
        '--rank',
        type=int,
        help='spawn single instance with given rank (as part of the '
             'rest of the cluster), instead of as multiple processes '
             'from a single instance'
    )
    parser.add_argument(
        '--agg-method',
        type=str,
        default='gradAllReduce',
        choices=tuple(TRAINERS.keys()),
        help='aggregation method used to aggregate gradients or params '
             'across all workers during training'
    )
    parser.add_argument(
        '--agg-period',
        type=int,
        default=1,
        help='if applicable, the period at which an aggregation occurs'
    )
    parser.add_argument(
        '--agg-prob',
        type=float,
        default=1.0,
        help='if applicable, the probability with which agg occurs'
    )
    parser.add_argument(
        '--weighted-gossip',
        action='store_true',
        help='applicable only to gossip-like protocols - weighted '
             'selection of peers based on distance in a topology '
             'generated randomly (2D-uniform)'
    )
    parser.add_argument(
        '--penalty',
        type=float,
        default=1,
        help='the penalty to apply to distances if using '
             '--weighted-gossip'
    )
    parser.add_argument(
        '--elastic-alpha',
        type=float,
        default=0.5,
        help='"moving rate" for elastic gossip'
    )
    parser.add_argument(
        '--momentum-correction',
        action='store_true',
        help='if applicable, correct for momentum during aggregation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='random seed'
    )
    parser.add_argument(
        '--master-loc',
        type=str,
        default='127.0.0.1:29500',
        help='<hostname>:<port> of rank-0th worker'
    )
    parser.add_argument(
        '--comm-backend',
        type=str,
        choices=('tcp', 'gloo'),
        default='tcp',
        help='communication backend between workers'
    )
    args = parser.parse_args(argv)
    args = set_default_args(args)
    args = init_log_path(args)
    args = checkpoint_check(args)
    args = parse_master_loc(args)
    args = gpu_check(args)
    args = trainer_check(args)
    args.timestamp = datetime.now().isoformat()
    return args


def get_gossip_weights(args):
    if not args.weighted_gossip:
        return args
    if args.num_workers < 2:
        return args
    args.node_locations, args.gossip_weights = \
        generate_topo_based_weights(args.num_workers, args.penalty)
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed(args.seed)
    args = get_gossip_weights(args)
    save_input_args_as_metadata(args)
    Experiment = EXPERIMENTS[args.experiment]
    Trainer = TRAINERS[args.agg_method]
    experiment = Experiment(args, Trainer)
    # Ideally this would use logging instead of print, but for some
    # reason it messes up other loggers used by the workers. This is
    # a workaround.
    print('Starting exp: {}'.format(args.exp_id))
    print('Logs at: {}'.format(args.log_path))
    experiment.run()
    print('Done exp: {}'.format(args.exp_id))
    return


if __name__ == '__main__':
    main()
