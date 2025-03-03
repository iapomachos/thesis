import numpy as np
import time

from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

start_time=time.time()

A = np.loadtxt(fname = "Susy100k", delimiter = ",")

X_train, X_test, y_train, y_test = train_test_split(A[:,1:18], A[:,0] ,test_size=0.4, random_state=0)

start_train_test_score_time=time.time()
y_train= y_train.astype(int)
y_test= y_test.astype(int)

svc1 = svm.SVC().fit(X_train,y_train)
svc1.score(X_test,y_test)

auROC = roc_auc_score(y_test, svc1.predict(X_test))

print(auROC)
elapsed_train_test_score_time = time.time() - start_train_test_score_time
elapsed_time = time.time() - start_time
print(elapsed_time)
print(elapsed_train_test_score_time)
