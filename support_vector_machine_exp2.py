from sklearn import datasets
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np

X,y=datasets.make_blobs(n_samples=50,centers=2,random_state=6)
print(X.shape,y.shape)

model=SVC(kernel='linear',C=1.0)
model.fit(X,y)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='inferno',s=100,edgecolors='k')
plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=150,edgecolors='k',facecolors='none',label='Support Vectors')
print('model co efficients',model.coef_)
print('model intercept',model.intercept_)
w=model.coef_[0]
b=model.intercept_[0]
x_range=np.linspace(X[:,0].min()-1,X[:,0].max()+1)
print(x_range)
y_boundary=-(w[0]*x_range+b)/w[1]
print(y_boundary)
plt.plot(x_range,y_boundary,'k-',label='Decision Boundary')
margin=1/np.linalg.norm(w)
print(margin)
y_margin_up=y_boundary+margin
y_margin_down=y_boundary-margin
plt.plot(x_range,y_margin_up,'k--',linewidth=1)
plt.plot(x_range,y_margin_down,'k--',linewidth=1)
plt.title('SVM decision boundary with support vectors')
plt.legend()
plt.grid(True)
plt.show()

