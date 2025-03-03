from sklearn.linear_model import Ridge
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

y =loadtxt("YearPredictionfirstcol")
X =loadtxt("YearPredictionrestcol",delimiter=',')


ridge = Ridge(alpha = 1.0,max_iter = 2000)
ridge.fit(X, y)
coef_ridge = ridge.coef_
coef_=ridge.coef_ * X + ridge.intercept_
plt.plot(X, coef_, 'g-', label="ridge regression")
plt.show()
