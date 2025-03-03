from sklearn.linear_model import Ridge
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import time

start_time=time.time()

y =loadtxt("fried_delvefirstcol")
X =loadtxt("fried_delverestcol",delimiter=',')

start_fit_time=time.time()
ridge = Ridge(alpha = 1.0,max_iter = 1000)
ridge.fit(X, y)
elapsed_fit_time = time.time() - start_fit_time


elapsed_time = time.time() - start_time
print(elapsed_time)
print(elapsed_fit_time)
