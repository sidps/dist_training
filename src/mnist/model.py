from torch import nn
from torch.nn import functional as F
from ..common.base_architecture import BaseArchitecture


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(28 * 28, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 1024)
        self.out = nn.Linear(1024, 10)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal(param)

    def forward(self, x):
        # input
        x = x.view(-1, 28 * 28)
        # reshaped
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.l1(x)
        x = F.relu(x)
        # l1
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.l2(x)
        x = F.relu(x)
        # l2
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.l3(x)
        x = F.relu(x)
        # l3
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.out(x)
        x = F.log_softmax(x, dim=1)
        # output
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
