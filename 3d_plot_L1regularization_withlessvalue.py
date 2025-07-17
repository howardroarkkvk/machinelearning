import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# L=1+w1^2+w2^2+5|w1| --> L=1+x^2+y^2+5|x|
w1=np.linspace(-4,4) # by default linspace creates 100 intervals it inlueds the boundaries...
w2=np.linspace(-4,4)


x,y=np.meshgrid(w1,w2)

z=1+x**2+y**2+0.5*np.abs(x)


z1=1+x**2+y**2

z2=1+x**2+y**2+0.1*np.abs(x+y)


fig=plt.figure(figsize=(10,8))

# row 1, col1 
ax=fig.add_subplot(2,2,1,projection='3d')
ax.plot_surface(x,y,z,cmap='coolwarm',edgecolor='none',alpha=0.9)
ax.set_xlabel('Weight W1',fontsize=10)
ax.set_ylabel('Weight W2',fontsize=10)
ax.set_zlabel('Weight L',fontsize=10)
ax.set_title('Loss Surface = L=1+w1^2+w2^2+.5|w1| ',fontsize=20)

# row 1, col2 
ax=fig.add_subplot(2,2,2,projection='3d')
ax.plot_surface(x,y,z1,cmap='viridis',edgecolor='none',alpha=0.9)
ax.set_xlabel('Weight W1',fontsize=10)
ax.set_ylabel('Weight W2',fontsize=10)
ax.set_zlabel('Weight L',fontsize=10)
ax.set_title('Loss Surface = L=1+w1^2+w2^2 ',fontsize=20)


ax=fig.add_subplot(2,2,3,projection='3d')
ax.plot_surface(x,y,z2,cmap='viridis',edgecolor='none',alpha=0.9)
ax.set_xlabel('Weight W1',fontsize=10)
ax.set_ylabel('Weight W2',fontsize=10)
ax.set_zlabel('Weight L',fontsize=10)
ax.set_title('Loss Surface = L=1+w1^2+w2^22+.5|w1+w2| ',fontsize=20)



plt.tight_layout()
plt.show()