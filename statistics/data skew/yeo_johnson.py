from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt


X = [-10, -5, 0, 1, 5, 50]
# Box-Cox
# PowerTransformer(method='box-cox')

# Yeo-Johnson
yj=PowerTransformer(method='yeo-johnson')
transformed=yj.fit_transform([[x] for x in X])
print(transformed)


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.hist(X,bins=6,color='blue',alpha=0.7)


plt.subplot(1,2,2)
plt.hist(transformed,bins=6,color='red',alpha=0.7)

plt.show()