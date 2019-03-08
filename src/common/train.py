""" Expected to follow tnt.engine """
import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from contextlib import ExitStack, contextmanager
from .util import save_checkpoint, load_checkpoint


class Trainer(object):
    """ Single process trainer. """

    def __init__(self, rank, data, architecture, args, logger):
        self.args = args
        self.rank = rank
        self.logger = logger
        self.logger.debug('Initializing trainer.')
        self.train_loader = data.get_train(self.rank)
        self.val_loader = data.get_val()
        self.test_loader = data.get_test()
        self.model = architecture.get_new_model()
        self.loss_fn = architecture.get_loss_fn()

        # These are hooks. on-forward is a misnomer and is called
        # after backward but before update.
        self.on_forward_fn = None
        self.on_update_fn = None

        # `start_epoch` is usually `0` except if we start from a
        # previously saved checkpoint.
        self.start_epoch = 0
        self.epoch = 0
        self.update = 0

        if self.args.gpu:
            # This splits models evenly across available GPUs
            self.device_id = self.rank % torch.cuda.device_count()
            with torch.cuda.device(self.device_id):
                self.model.cuda()
            self.logger.debug('Model on GPU %s' % self.device_id)

        self._init_optimizer()

        if self.args.from_checkpoint:
            self.load_checkpoint()

        self.logger.info('Initialized trainer')

    def _init_optimizer(self):
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum,
            nesterov=self.args.nesterov,
            weight_decay=self.args.weight_decay,
        )

        # This scheduler anneals the learning rate with the given
        # schedule and annealing factor
        self.lr_scheduler = None
        if self.args.anneal_milestones:
            self.lr_scheduler = MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.args.anneal_milestones,
                gamma=self.args.anneal_factor,
            )

    def train(self):
        self.logger.info('Starting training')

        for epoch in range(self.start_epoch + 1, self.args.epochs + 1):
            self.epoch = epoch
            self.logger.debug('Starting epoch %s' % self.epoch)
            for input_data, target in self.train_loader:
                self.update += 1
                self.train_step(input_data, target)
            # For now, we only checkpoint at the end of an epoch
            # rather than at an update
            if self.args.checkpoint_interval and \
                    self.epoch % self.args.checkpoint_interval == 0:
                self.save_checkpoint()
            # Increment the learning rate scheduler which keeps track of
            # annealing:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.logger.info('Finished training')

        # Compute test-set accuracy and log
        acc = self.evaluate()
        self.logger.info({
            'Rank': self.rank,
            'TestAcc': acc,
        })
        self.logger.info({
            'Rank': self.rank,
            'TestAccPerc': '%.2f %%' % (acc * 100),
        })
        return

    def train_step(self, input_data, target):
        self.logger.debug('Starting update %s' % self.update)

        # Set mode to train:
        self.model.train()

        # Transfer data to GPU
        if self.args.gpu:
            with torch.cuda.device(self.device_id):
                input_data = input_data \
                    .cuda(self.device_id, async=self.args.async_cuda)
                target = target \
                    .cuda(self.device_id, async=self.args.async_cuda)
            self.logger.debug('Loaded data to GPU %s' % self.device_id)

        # Compute loss and gradients for the batch
        x = Variable(input_data)
        y = Variable(target)
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        train_loss = loss.data[0]

        self.logger.debug('Computed backward.')
        # This is where we potentially aggregate gradients across
        # workers
        if self.on_forward_fn:
            self.on_forward_fn()

        # Gradient descent step
        self.optimizer.step()
        self.logger.debug('Executed optim step.')

        # This is where we potentially aggregate parameters across
        # workers
        if self.on_update_fn:
            self.on_update_fn()

        # Log some metrics
        if self.update % self.args.log_interval == 0:
            val_loss, val_acc = self.evaluate(val=True)
            self.logger.info({
                'Rank': self.rank,
                'Epoch': self.epoch,
                'Update': self.update,
                'TrainLoss': train_loss,
            })
            self.logger.info({
                'Rank': self.rank,
                'Epoch': self.epoch,
                'Update': self.update,
                'ValLoss': val_loss,
            })
            self.logger.info({
                'Rank': self.rank,
                'Epoch': self.epoch,
                'Update': self.update,
                'ValAcc': val_acc,
            })
            self.logger.info({
                'Rank': self.rank,
                'Epoch': self.epoch,
                'Update': self.update,
                'ValAccPerc': '%.2f %%' % (val_acc * 100),
            })

        self.logger.debug('Finished update %s' % self.update)
        return

    @contextmanager
    def _maybe_on_cpu_for_eval(self):
        with ExitStack() as stack:
            if self.args.gpu:
                stack.enter_context(torch.cuda.device(self.device_id))
                if not self.args.eval_on_gpu:
                    self.model.cpu()
            self.logger.debug('On CPU for eval')
            yield
            if self.args.gpu and not self.args.eval_on_gpu:
                self.model.cuda()
            self.logger.debug('Off CPU after eval')

    def evaluate(self, val=False):
        """
            Evaluate all batches of validation or train set and aggregate
        """
        with self._maybe_on_cpu_for_eval():
            loader = self.val_loader if val else self.test_loader
            results = [self.evaluate_batch(input_data, target, val=val)
                       for input_data, target in loader]
            if not val:
                return sum(results) / len(results)

            losses, accs = list(zip(*results))
            return sum(losses) / len(losses), \
                sum(accs) / len(accs)

    def evaluate_batch(self, input_data, target, val=False):
        """ Evaluate single mini batch of train or validation set."""
        self.optimizer.zero_grad()
        self.model.eval()

        if self.args.gpu and self.args.eval_on_gpu:
            with torch.cuda.device(self.device_id):
                input_data = input_data \
                    .cuda(self.device_id, async=self.args.async_cuda)
                target = target \
                    .cuda(self.device_id, async=self.args.async_cuda)

        # Compute outputs and compare with targets
        x = Variable(input_data, volatile=True)
        out = self.model(x)

        _, pred = out.data.max(1, keepdim=True)
        acc = pred.eq(target.view_as(pred)).float().cpu().numpy().mean()

        if not val:
            return acc

        # If validation, then compute loss as well
        y = Variable(target, volatile=True)
        loss = self.loss_fn(out, y)
        val_loss = loss.data[0]

        return val_loss, acc

    def _get_lr_scheduler_state(self):
        """ Hack for 0.3.1; 0.4.0 onwards have state_dict and
            load_state_dict for lr scheduler.
        """
        if self.lr_scheduler is None:
            return None
        return {
            key: value for key, value in self.lr_scheduler.__dict__.items()
            if key != 'optimizer'
        }

    def _set_lr_scheduler_state(self, state):
        """ Hack for 0.3.1; 0.4.0 onwards have state_dict and
            load_state_dict for lr scheduler.
        """
        if state is None:
            return
        assert self.lr_scheduler is not None
        self.lr_scheduler.__dict__.update(state)
        return

    def get_state(self):
        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if self.args.gpu:
            cuda_rng_state = torch.cuda.get_rng_state(self.device_id)
        self.model.eval()
        state = dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self._get_lr_scheduler_state(),
            epoch=self.epoch,
            update=self.update,
            rank=self.rank,
            rng_state=rng_state,
            cuda_rng_state=cuda_rng_state,
        )
        return state

    def set_state(self, state):
        assert self.rank == state['rank']
        self.start_epoch = state['epoch']
        self.update = state['update']
        torch.set_rng_state(state['rng_state'])
        if self.args.gpu:
            torch.cuda.set_rng_state(
                state['cuda_rng_state'],
                device=self.device_id,
            )

        with ExitStack() as stack:
            if self.args.gpu:
                stack.enter_context(torch.cuda.device(self.device_id))
            self.model.load_state_dict(state['model'])
            # Because we apparently have to:
            # https://github.com/pytorch/pytorch/issues/2830#issuecomment-336183179
            self._init_optimizer()
            self.optimizer.load_state_dict(state['optimizer'])
            if self.args.gpu:
                for opt_state in self.optimizer.state.values():
                    for k, v in opt_state.items():
                        if torch.is_tensor(v):
                            opt_state[k] = v.cuda(self.device_id)
            if state['scheduler'] is not None:
                assert self.lr_scheduler is not None
                self._set_lr_scheduler_state(state['scheduler'])

        self.model.eval()
        self.logger.debug('Completed setting state from dict.')
        return

    def save_checkpoint(self):
        self.logger.info('Saving checkpoint')
        state = self.get_state()
        save_checkpoint(self.args.log_path, self.rank, state)
        return

    def load_checkpoint(self):
        self.logger.info('Loading from checkpoint')
        state = load_checkpoint(self.args.log_path, self.rank)
        self.set_state(state)
        return
