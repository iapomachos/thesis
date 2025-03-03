import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


A = np.loadtxt(fname = "Susy500k", delimiter = ",")
X = A[:,1:18]
y = A[:,0]

y = y.astype(int)

print(y)

svc1 = svm.SVC()

svc1.fit(X, y)

targets = svc1.predict(X)

print("SVC")
print("Accuracy: ", accuracy_score(y, targets))
print("Precision: ", precision_score(y, targets))
print("Recall: ", recall_score(y, targets))

