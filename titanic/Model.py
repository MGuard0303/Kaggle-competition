from torch import nn


class Vanilla(nn.Module):
    def __init__(self):
        super().__init__()

        self.fcs = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.loss_func = nn.BCELoss()

    def forward(self, x):
        x = self.fcs(x)

        return x
