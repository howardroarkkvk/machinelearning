import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)
data = np.random.exponential(scale=2,size=1000)
# print(data)

df=pd.DataFrame({'feature':data})
print(df)

skew_value=df['feature'].skew()
print(skew_value)

plt.figure(figsize=(12,10))

plt.subplot(3,2,1)
sns.histplot(df['feature'],kde=True)
plt.title('Histogram + KDE')

plt.subplot(3,2,2)
sns.boxplot(x=df['feature'])
plt.title('box plot')

plt.subplot(3,2,3)
sns.kdeplot(df['feature'],fill=True)
plt.title('kde plot')


plt.show()