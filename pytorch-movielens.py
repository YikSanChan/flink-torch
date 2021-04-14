#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import torch
from torch import nn


# In[2]:


n_users, n_items = 943, 1682
BATCH_SIZE = 32


# In[3]:


COLS = ["user_id", "movie_id", "rating", "timestamp"]
train_data = (
    pd.read_csv("ml-100k/u1.base", sep="\t", names=COLS)
    .drop(columns=["timestamp"])
    .astype(int)
)
test_data = (
    pd.read_csv("ml-100k/u1.test", sep="\t", names=COLS)
    .drop(columns=["timestamp"])
    .astype(int)
)


# In[4]:


print(train_data.head())


# In[5]:


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


model = MatrixFactorization(n_users, n_items)
print(model)


# In[6]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss()


# In[7]:


def train(data, model, loss_fn, optimizer):
    size = len(data)
    for batch in range(len(data) // BATCH_SIZE):
        df = train_data.sample(frac=BATCH_SIZE / len(data))
        users = df.user_id.values
        items = df.movie_id.values
        targets = torch.FloatTensor(df.rating.values)
        assert users.shape == (BATCH_SIZE,) == items.shape

        preds = model(users, items)
        loss = loss_fn(preds, targets)
        assert preds.shape == targets.shape

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 500 == 0:
            loss_val, current = loss.item(), batch * len(df)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")


# In[8]:


epochs = 5
for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    train(train_data, model, loss_fn, optimizer)
print("Done!")


# In[9]:


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


# In[11]:


model = MatrixFactorization(n_users, n_items)
model.load_state_dict(torch.load("model.pth"))


# In[12]:


model.eval()
sample_record = test_data.iloc[0]
x_user, x_item, y = (
    sample_record["user_id"],
    sample_record["movie_id"],
    sample_record["rating"],
)


# In[13]:


with torch.no_grad():
    pred = model([x_user], [x_item])
    print(f'Predicted: "{pred}", Actual: "{y}"')
