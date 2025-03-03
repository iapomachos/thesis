from sklearn.metrics import mean_squared_error
from numpy import *
import numpy as np


scikit = loadtxt("scikit.txt")
spark = loadtxt("spark.txt")

mse = mean_squared_error(scikit.astype(int),spark.astype(int))

print(mse)
