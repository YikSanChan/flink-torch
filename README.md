# Flink Torch

I want to explore if we can leverage PyFlink to run ML batch prediction performant. Here we go.

## What're the files?

- `ml-100k/`: [Movielens 100K](https://grouplens.org/datasets/movielens/100k/) dataset. It is pretty much the "Hello world" in recsys world.
- `model_def.py`: A naive Matrix Factorization implementation, shamelessly copied from [Kaggle](https://www.kaggle.com/shihabshahriar/pytorch-movielens/data), and polished following [PyTorch quickstart](https://github.com/pytorch/tutorials/blob/master/beginner_source/basics/quickstart_tutorial.py).
- `pytorch_movielens.py`: It trains the model, saves it, loads it back and runs a prediction.
- `model.pth`: A saved model from running `pytorch_movielens.py`.
- `udf_def.py`: PyFlink UDF that loads the model againsts which it runs batch prediction.
- `test_udf_predict.py`: Unit test of the PyFlink UDF.

## Try it!

Steps:
1. Make sure you have `conda` installed.
1. Create a conda env `flink-ml` with `conda env create -f environment.yml`, and activate with `conda activate flink-ml`.
1. (Optional) Re-use the existing trained model `model.pth`, or re-train the model with `python pytorch_movielens.py`.
1. Run batch prediction with `python batch_prediction.py`. You should find predictions in your standard out. It says, user 269 may love movies `[316,486,664,729,1020]` best, and user 270 may love movies `[77,83,306,800,1014]` best.

```
...
5> +U(269,316,486,664,729,1020)
5> +U(270,77,83,306,800,1014)
```
