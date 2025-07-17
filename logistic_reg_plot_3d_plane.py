import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit


np.random.seed(42)
X1=np.random.uniform(-3,3,100)
X2=np.random.uniform(-3,3,100)
X=np.column_stack((X1,X2))
print(X)

y=(X1+X2+np.random.rand(100))>0
print(y)

y=y.astype(int)
print(y)

model=LogisticRegression()
model.fit(X,y)

# Generate grid for plotting
x1_range=np.linspace(-3,3,50)
x2_range=np.linspace(-3,3,50)

x1_grid,x2_grid=np.meshgrid(x1_range,x2_range)
print(x1_grid,x2_grid)

z=model.coef_[0][0]*x1_grid + model.coef_[0][1]*x2_grid+model.intercept_[0]
print(z)

y_prob=expit(z)
print(y_prob)

fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111,projection='3d')
ax.set_xlabel('x1')
ax.set_ylabel('y1')
ax.set_zlabel('predicted probability')
ax.set_title("3D logistic regression surface")
ax.plot_surface(x1_grid,x2_grid,y_prob,cmap='coolwarm',alpha=0.7,edgecolor='none')
ax.scatter(X1,X2,c=y,cmap='bwr',edgecolors='k',label='Data')

plt.legend()
plt.show()