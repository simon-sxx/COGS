import torch

class Algorithm(torch.nn.Module):

    def __init__(self, args):
        super(Algorithm, self).__init__()

    def update(self, x, y, **kwargs):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)