from torch import nn
from torch.nn import functional as F
from ..common.base_architecture import BaseArchitecture


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class Architecture(BaseArchitecture):
    def __init__(self):
        super(Architecture, self).__init__()

    @classmethod
    def get_new_model(cls):
        return Net()

    @classmethod
    def get_loss_fn(cls):
        return F.nll_loss
