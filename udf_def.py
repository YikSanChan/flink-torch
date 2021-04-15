from pyflink.table.udf import udf
from pyflink.table import DataTypes

import torch

from model_def import MatrixFactorization


@udf(result_type=DataTypes.DOUBLE())
def predict(user, item):
    n_users, n_items = 943, 1682
    model = MatrixFactorization(n_users, n_items)
    model.load_state_dict(torch.load("model.pth"))
    # print(model)
    return model([user], [item]).item()
