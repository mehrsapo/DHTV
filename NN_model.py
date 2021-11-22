import torch
import torch.nn as nn
import torch.nn.functional as functional


class Net(nn.Module):
    """
    input size :
    N x 2 (point in space)

    Output size of each layer:
    N x 2 -> fc1 -> N x h
          -> fc2 -> N x h
          -> fc3 -> N x h
          -> fc4 -> N x h
          -> fc5 -> N x h
          -> fc6 -> N x h
          -> fc7 -> N x h
          -> fc8 -> N x h
          -> fc9 -> N x h
          -> fc10 -> N x h
          -> fc11 -> N x 1
    """

    def __init__(self, dimension, two_layer, hidden):

        super().__init__()
        self.hidden = hidden  # number of hidden neurons
        self.two_layer = two_layer
        self.fc1 = nn.Linear(dimension, hidden)
        if two_layer is False:
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, hidden)
            self.fc4 = nn.Linear(hidden, hidden)
            self.fc5 = nn.Linear(hidden, hidden)
        # self.fc6 = nn.Linear(hidden, hidden)
        # self.fc7 = nn.Linear(hidden, hidden)
        # self.fc8 = nn.Linear(hidden, hidden)
        # self.fc9 = nn.Linear(hidden, hidden)
        # self.fc10 = nn.Linear(hidden, hidden)
        self.fc6 = nn.Linear(hidden, 1)

        self.num_params = self.get_num_params()

    def get_num_params(self):
        num_params = 0
        for param in self.parameters():
            num_params += torch.numel(param)

        return num_params

    def forward(self, x):
        """ """
        x = functional.relu(self.fc1(x))
        if self.two_layer is False:
            x = functional.relu(self.fc2(x))
            x = functional.relu(self.fc3(x))
            x = functional.relu(self.fc4(x))
            x = functional.relu(self.fc5(x))
        # x = functional.relu(self.fc6(x))
        # x = functional.relu(self.fc7(x))
        # x = functional.relu(self.fc8(x))
        # x = functional.relu(self.fc9(x))
        # x = functional.relu(self.fc10(x))
        x = self.fc6(x).squeeze(1)

        return x
