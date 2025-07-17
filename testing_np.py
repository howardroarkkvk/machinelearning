import numpy as np
from sklearn.linear_model import LogisticRegression
# x=np.linspace(10,20,15) # inclusive of boundaries....it will give 30 values...
# print(x,x.shape)
# p,q=np.meshgrid(x,x)
# print(p,p.shape)
# print(q,q.shape)
# print(p.ravel())
# print(q.ravel())
# t=np.c_[p.ravel(),q.ravel()]
# print(t,t.shape)

# a=np.array([[0.1,.5],[.5,0.1],[.9,.9],[.1,.1]])
# b=np.array([0,1,0,0])
# # print(a)
# # t=a-0.5
# # print(t)
# model=LogisticRegression()
# model.fit(a,b)
# y_p=model.predict(a)
# print(y_p)
# y_prob=model.predict_proba(a)
# print(y_prob)
# correct_idx=np.where(y_p==b)[0]
# print(correct_idx)

# print(y_prob[correct_idx])
# print(y_prob[correct_idx]-.5)
# print(np.argsort(np.abs(y_prob[correct_idx]-.5)))


x=np.linspace(-4,4,200)
mesh_x,mesh_y=np.meshgrid(x,x)
print(x)
print(mesh_x)
print(mesh_y)
