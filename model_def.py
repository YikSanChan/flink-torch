import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)

    def forward(self, user, item):
        user = torch.LongTensor(user) - 1
        item = torch.LongTensor(item) - 1
        u, it = self.user_factors(user), self.item_factors(item)
        x = (u * it).sum(1)
        assert x.shape == user.shape
        return x * 5
