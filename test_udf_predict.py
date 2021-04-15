from udf_def import predict


def test_predict():
    f = predict._func
    pred = f(1, 2)
    assert type(pred) is float
