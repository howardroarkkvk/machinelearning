import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X=np.array([[90,18],[130,37],[95,19],
            [140,35],[92,20],[145,34] ])

y=np.array([-1,-1,-1,1,1,1])

model=SVC(kernel='rbf',gamma=0.1)
model.fit(X,y)

def plot_boundary(X,y,model):
    x_min,x_max=X[:,0].min()-10,X[:,0].max()+10
    # print(x_min,x_max)
    y_min,y_max=X[:,1].min()-5,X[:,1].max()+5
    # print(y_min,y_max)
    xx,yy=np.meshgrid(np.linspace(x_min,x_max,300),np.linspace(y_min,y_max,300))
    print(xx.shape,yy.shape)
   
    t=model.predict(np.c_[xx.ravel(),yy.ravel()])
    print(t,t.shape)

    k=t.reshape(xx.shape)
    print(k,k.shape)
    print(np.unique(k))
    print(model.support_vectors_[:,0])
    print(model.support_vectors_[:,1])

    plt.contour(xx,yy,k,levels=[-1,1],cmap='coolwarm',alpha=0.9,linestyles=['--','-'])

    plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr',edgecolors='k',s=100)
    # plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=150,linewidths=1.5,facecolors='green',edgecolors='k',label='support vectors')
    plt.xlabel('Blood sugar')
    plt.ylabel('BMI')
    plt.title('SVC with RBF kernel - Non linear Boundary')
    plt.grid(True)
    plt.show()


plot_boundary(X,y,model)