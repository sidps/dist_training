from .data import Data
from .model import Architecture
from ..common.cluster import Cluster
from ..common.util import setup_logger, teardown_logger


class Experiment(object):
    def __init__(self, cli_args, TrainerClass):
        self.args = cli_args
        self.TrainerClass = TrainerClass
        self.data = Data(self.args)
        self.cluster = Cluster(self.args)
        self.trainer_states = []

    def _setup_trainer(self, rank, logger):
        return self.TrainerClass(
            data=self.data,
            architecture=Architecture,
            rank=rank,
            args=self.args,
            logger=logger,
        )

    def _work(self, rank):
        train_logger = setup_logger(
            log_path=self.args.log_path,
            log_level=self.args.log_level,
            rank=rank,
            append=self.args.from_checkpoint,
        )
        trainer = self._setup_trainer(rank, train_logger)
        trainer.train()
        trainer.evaluate()
        teardown_logger(train_logger)
        return

    def run(self):
        if self.args.num_workers == 1:
            # Single node special case because torch dist
            # cannot handle this
            self._work(rank=0)
        elif self.args.rank is not None:
            # Case where we only want to run one worker using this
            # process in the distributed system.
            self.cluster.run_single(
                self.args.rank,
                self._work,
            )
        else:
            # Case where we use this process to spawn multiple
            # sub-processes, each handling a worker
            self.cluster.run_processes(
                self._work,
            )
        return
