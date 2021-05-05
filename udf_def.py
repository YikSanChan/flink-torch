from pyflink.table.udf import udf, ScalarFunction
from pyflink.table import DataTypes

import torch
import pandas as pd

from model_def import MatrixFactorization


class Predict(ScalarFunction):
    def open(self, function_context):
        n_users, n_items = 943, 1682
        model = MatrixFactorization(n_users, n_items)
        model.load_state_dict(torch.load("model.pth"))
        model.eval()
        self.model = model

    def eval(self, users, items):
        with torch.no_grad():
            preds = self.model(users, items)
            return pd.Series(preds)

predict = udf(Predict(), result_type=DataTypes.DOUBLE(), func_type="pandas")
