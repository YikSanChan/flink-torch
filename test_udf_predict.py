import pandas as pd

from udf_def import predict


def test_predict():
    f = predict._func
    users = pd.Series([1, 2, 3])
    items = pd.Series([1, 4, 9])
    preds = f(users, items)
    assert isinstance(preds, pd.Series)
    assert len(preds) == 3
