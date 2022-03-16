import torch
import torch.nn as nn
import torch.nn.functional as functional


class Net(nn.Module):
    """
    D Fully-connected network with ReLU activations.
    input size: N x D (point in space)
    Network (layer type -> output size):
    fc1    -> N x h
    fc2    -> N x h
    ...
    fcL    -> N x h  (L=layer)
    fclast -> N x 1

    """

    def __init__(self, dimension, layer, hidden):

        super().__init__()
        self.num_hidden_layers = layer
        self.num_hidden_neurons = hidden

        self.fc1 = nn.Linear(dimension, self.num_hidden_neurons)
        self.fchidden = nn.ModuleList(
            [nn.Linear(self.num_hidden_neurons, self.num_hidden_neurons)
             for i in range(self.num_hidden_layers - 1)]
        )
        self.fclast = nn.Linear(self.num_hidden_neurons, 1)

        self.num_params = self.get_num_params()

    def get_num_params(self):
        num_params = 0
        for param in self.parameters():
            num_params += torch.numel(param)

        return num_params

    def forward(self, x):
        """ """
        x = functional.relu(self.fc1(x.double()))

        for fclayer in self.fchidden:
            x = functional.relu(fclayer(x.double()))

        x = self.fclast(x).squeeze(1)

        return x
