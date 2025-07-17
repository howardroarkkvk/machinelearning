import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from numpy.linalg import norm

X,y=make_classification(n_samples=100,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,random_state=2)
print(X,y)


model= LogisticRegression()
model.fit(X,y)
y_pred=model.predict(X)
y_ypred=np.c_[y,y_pred]
print(y_ypred)
y_pred_prob=model.predict_proba(X)
print('predict proba', y_pred_prob[:,0]) # it gives the output in p,1-p format

correct_idx=np.where(y_pred==y)[0]
print('correct_index')
print(correct_idx)

# label 1 which are close to boundary....
print(np.abs(y_pred_prob[:,0][correct_idx]-0.5))
close_to_boundary=correct_idx[np.argsort(np.abs(y_pred_prob[:,0][correct_idx]-0.5))[0:5]]
print(close_to_boundary)

# label 1 which are far from the boundary
far_from_boundary=correct_idx[np.argsort(np.abs(y_pred_prob[:,0][correct_idx]-0.5))[-5:]]
print(far_from_boundary)

# flip their lables to make them misclassfied
y[close_to_boundary]=1-y[close_to_boundary]
print(y[close_to_boundary])
y[far_from_boundary]=1-y[far_from_boundary]
print(y[far_from_boundary])


# # label 1 which are close to boundary....
# print(np.abs(y_pred_prob[:,1][correct_idx]-0.5))
# close_to_boundary=correct_idx[np.argsort(np.abs(y_pred_prob[:,1][correct_idx]-0.5))[0:2]]
# print(close_to_boundary)

# # label 1 which are far from the boundary
# far_from_boundary=correct_idx[np.argsort(np.abs(y_pred_prob[:,1][correct_idx]-0.5))[-2:]]
# print(far_from_boundary)

# # flip their lables to make them misclassfied
# y[close_to_boundary]=1-y[close_to_boundary]
# print(y[close_to_boundary])
# y[far_from_boundary]=1-y[far_from_boundary]
# print(y[far_from_boundary])




y_pred=model.predict(X)

plt.figure(figsize=(8,6))
x1_min,x1_max=X[:,0].min(),X[:,0].max()
x2_min,x2_max=X[:,1].min(),X[:,1].max()
# print(x1_min,x1_max)
# print(x2_min,x2_max)
xx,yy=np.meshgrid(np.linspace(x1_min,x1_max,200),np.linspace(x2_min,x2_max,200))
# print(xx,yy)

# for every cor-ordinate i.e x it will concatenate with y and for all the co-ordinates formed it will predict the values
flattened_values=np.c_[xx.ravel(),yy.ravel()]
print(flattened_values,flattened_values.shape)

z=model.predict(flattened_values)
z=z.reshape(xx.shape)
print(z,z.shape)

#plt decision boundary
plt.contourf(xx,yy,z,alpha=0.3,cmap='coolwarm')


w=model.coef_[0]
w=np.array([w])
print('w is :',w)
b=model.intercept_[0]
print('intercept is ',b)
print('shapes')
# print(X.shape,w.shape,b.shape)
def distance_from_boundary(x,w,b,idx,true_label,correct):
    # print(x,x.shape)
    # print('idx value is ',idx)
    # print(w.shape,x[idx,:].T.shape,b.shape)
    z=np.abs(np.dot(w,x[idx,:].T)+b)/norm(w)
    sigmoid=1/(1+np.exp(-z))
    if true_label==1 and correct:
        loss=-np.log2(sigmoid)
    elif true_label==1 and not correct:
        loss=-np.log2(1-sigmoid)
    elif true_label==0 and correct:
        loss=-np.log2(sigmoid)
    elif true_label==0 and not correct:
        loss=-np.log2(1-sigmoid)
    print('loss is:',loss)
    return loss.reshape(-1)


#1/1+e power -z



for true_label,marker,label_name in zip([0,1],['o','^'],['class 0','class1']):
    print('true label is:',true_label)
    idx=np.where(y==true_label)[0]
    correct=idx[y_pred[idx]==y[idx]]
    print('correct length',len(correct))
    incorrect=idx[y_pred[idx]!=y[idx]]
    print('incorrect length',len(incorrect))

    correct_1=True
    correct_distances=distance_from_boundary(X,w,b,correct,true_label,correct_1)
    print('correct distances is:',correct_distances,correct_distances.shape)

    incorrect_1=False
    incorrect_distances=distance_from_boundary(X,w,b,incorrect,true_label,incorrect_1)
    print('incorrect ones distances is:',incorrect_distances,incorrect_distances.shape)

    plt.scatter(X[correct,0],X[correct,1],c='green',marker=marker,edgecolors='k',label=f'{label_name} - correct')
    # for i in range(0,len(correct)):
    #     plt.text(X[correct[i],0]+0.05,X[correct[i],1],f"{correct_distances[i]:.1f},",fontsize=7,color='black')
    plt.scatter(X[incorrect,0],X[incorrect,1],c='red',marker=marker,edgecolors='k',label=f'{label_name} - In correct')
    # for i in range(0,len(incorrect)):
    #     plt.text(X[incorrect[i],0]+0.05,X[incorrect[i],1],f"{incorrect_distances[i]:.1f},",fontsize=7,color='black')
    

plt.title('Logisitc regression classification with misclassificatoin')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()