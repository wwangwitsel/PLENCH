import torch
import torch.nn as nn
import torch.nn.functional as F
from plench.lib import resnet

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.n_outputs = hidden_dim
        #self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        #print(out.dtype)
        out = self.fc1(out)
        out = self.relu1(out)
        #out = self.fc2(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def calc_dim(input_shape):
    input_shape = input_shape[1:]
    num_features = 1
    for s in input_shape:
        num_features *= s
    return num_features   

def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    dim = calc_dim(input_shape)
    if hparams["model"] == "MLP":
        return MLP(dim, 500)
    elif hparams["model"] == "LeNet":
        return LeNet()
    elif hparams["model"] == "ResNet":
        return resnet.resnet(depth=32)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features):
    return torch.nn.Linear(in_features, out_features)
