import torch
from collections import OrderedDict
from functools import partial
from pytest import fixture
from src.common.train import Trainer


class TestTrain(object):
    def on_forward_fn(self, trainer_ins):
        self.state['on_forward'] = repr(trainer_ins.model.state_dict())
        return

    def on_update_fn(self, trainer_ins):
        self.state['on_update'] = repr(trainer_ins.model.state_dict())
        return

    @fixture
    def sample_trainer(self, sample_data, sample_architecture, 
                       mock_args, test_logger):
        trainer = Trainer(
            rank=0, data=sample_data, architecture=sample_architecture,
            args=mock_args, logger=test_logger
        )
        self.state = OrderedDict()
        trainer.on_forward_fn = partial(
            self.on_forward_fn,
            trainer_ins=trainer
        )
        trainer.on_update_fn = partial(
            self.on_update_fn,
            trainer_ins=trainer
        )
        yield trainer
        trainer.on_forward_fn = None
        trainer.on_update_fn = None
        return

    def test_train_step(self, sample_trainer):
        # is this the best way to check state / changes?
        inputs, target = next(iter(sample_trainer.train_loader))
        self.state['initial'] = repr(sample_trainer.model.state_dict())
        sample_trainer.train_step(inputs, target)
        self.state['final'] = repr(sample_trainer.model.state_dict())

        assert list(self.state.keys()) == \
            ['initial', 'on_forward', 'on_update', 'final']
        assert self.state['initial'] == self.state['on_forward']
        assert self.state['on_update'] == self.state['final']
        assert self.state['initial'] != self.state['final']
        return

    @staticmethod
    def get_state(trainer):
        trainer.model.eval()
        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if trainer.args.gpu:
            cuda_rng_state = torch.cuda.get_rng_state(
                trainer.device_id)
        return dict(
            model=trainer.model.state_dict(),
            optimizer=trainer.optimizer.state_dict(),
            scheduler=trainer._get_lr_scheduler_state(),
            epoch=trainer.epoch,
            update=trainer.update,
            rank=trainer.rank,
            rng_state=rng_state,
            cuda_rng_state=cuda_rng_state,
        )

    def test_checkpointing(self, sample_trainer):
        sample_trainer.train()
        sample_trainer.save_checkpoint()
        original_state = repr(self.get_state(sample_trainer))
        assert original_state == repr(sample_trainer.get_state())
        sample_trainer.train()
        intermediate_state = repr(self.get_state(sample_trainer))
        assert intermediate_state != original_state
        sample_trainer.load_checkpoint()
        new_state = repr(self.get_state(sample_trainer))
        assert original_state == new_state
        return
