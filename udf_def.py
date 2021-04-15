from pyflink.table.udf import udf
from pyflink.table import DataTypes

import torch
import pandas as pd

from model_def import MatrixFactorization


@udf(result_type=DataTypes.DOUBLE(), func_type="pandas")
def predict(users, items):
    n_users, n_items = 943, 1682
    model = MatrixFactorization(n_users, n_items)
    model.load_state_dict(torch.load("model.pth"))
    return pd.Series(model(users, items).detach().numpy())
