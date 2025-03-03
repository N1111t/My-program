from torch import nn

class DNNModel(nn.Module):
    def __init__(self, input_size=768, num_classes=4, hidden_sizes=[256, 128]):
        super().__init__()
        self.relu = nn.ReLU()
        layers = [nn.Linear(input_size, hidden_sizes[0]), self.relu]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(self.relu)
        self.network = nn.Sequential(*layers)
        self.post_net = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        x = self.network(x)
        x = self.post_net(x)
        return x