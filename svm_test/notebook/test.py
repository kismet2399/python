from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
data = load_breast_cancer()
X = data.data
print(X.shape) # (569,30)
y = data.target
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()