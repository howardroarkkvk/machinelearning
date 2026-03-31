from scipy.stats import boxcox
import matplotlib.pyplot as plt


price = [50000, 60000, 80000, 120000, 500000]

price_transformed, lambda_ = boxcox(price)
print(price_transformed,lambda_)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.hist(price,bins=5,color='blue',alpha=0.7)


plt.subplot(1,2,2)
plt.hist(price_transformed,bins=5,color='red',alpha=0.7)

plt.show()

# λ value	Transformation
# 1	No change
# 0.5	Square root
# 0	Log
# -1	Reciprocal