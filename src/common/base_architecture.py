class BaseArchitecture(object):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_new_model(cls):
        raise NotImplementedError

    @classmethod
    def get_loss_fn(cls):
        raise NotImplementedError
