import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from numpy import *


np.set_printoptions(threshold=np.inf)

start_time=time.time()
X = loadtxt("CupBio.txt")
centers = loadtxt("CupBioCenters.txt")


fit_and_predict_start_time=time.time()
kmeans = KMeans(n_clusters = 20, n_init = 1000 , init = centers)
kmeans.fit(X)
predictions = kmeans.predict(X)
fit_and_predict_elapsed_time=time.time() - fit_and_predict_start_time

np.set_printoptions(threshold=np.inf)
print(predictions)


np.savetxt('scikit.txt',predictions)

elapsed_time = time.time() - start_time
print(elapsed_time)
print(fit_and_predict_elapsed_time)

