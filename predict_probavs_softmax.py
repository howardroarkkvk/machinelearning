import numpy as np
from sklearn.linear_model import LogisticRegression

X=np.array([[1],[2],[3],[4]])
y=np.array([0,0,1,1])

model=LogisticRegression()
model.fit(X,y)

print(model.coef_[0][0])
print(model.intercept_[0])

print(X.flatten())

z=X*model.coef_[0][0]+model.intercept_[0]
print('z value is',z)

func=lambda z:1/(1+np.exp(-z))
p_manual=func(z)
print('p manual is :')
print(func(z))

softmax_formula=lambda p:np.exp(p)/(np.exp(0)+np.exp(p))

# for binary classification sigmod becomes softmax if in softmax if we have 2 values as 0,p, so based on this the probabilities are calculated....
for i in range(0,len(X)):
    print(f"sample {X[i][0]} -- class 1 ---> {softmax_formula(z[i][0])}")

y_pred_proba=model.predict_proba(X)
print(y_pred_proba)

for i in range(len(X)):
    print(f"sample {X[i][0]}: [class 0:{1-p_manual[i]}, class 1: {p_manual[i]}] ")
