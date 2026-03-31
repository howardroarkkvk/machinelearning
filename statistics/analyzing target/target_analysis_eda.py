import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

path=r'E:\samples'

df=pd.read_csv(os.path.join(path,'Housing.csv'))
print(df)

print(df.info())
# we can figure out if there is any right skew or not using mean value > than median
# also if the max value is higher than the mean, then outliers exist
print(df['price'].describe())

# we can see the same in plot using histplot
plt.figure(figsize=(12,8))

plt.subplot(3,3,1)
sns.histplot(df['price'],kde=True)
plt.title('Target Distribution - prices')
# plt.show()

plt.subplot(3,3,2)
sns.boxplot(x=df['price'])
plt.title('Box Plot - prices')
# plt.show()

# checking skewness as skew value is greater than 1 it is right skew...0 normal and -1 highly left skewed
print(df['price'].skew())

# finding outliers

q1=df['price'].quantile(0.25)
q3=df['price'].quantile(0.75)
iqr=q3-q1
lower=q1-1.5*iqr
higher=q3+1.5*iqr

print(f'Q1-{q1} \n Q3-{q3} \n iqr - {iqr} \n lower - {lower} \n higher - {higher} \n')
outliers=df[(df['price']< lower) | (df['price']>higher)]
print(f'outliers - {outliers} \n, length - {len(outliers)}')

# transform target using log transformation as it is right skewed

df['price_log']=np.log(df['price'])
print(df)

print('value counts')
print(df['basement'].value_counts(normalize=True))

# plt.figure(figsize=(12,8))

plt.subplot(3,3,4)
sns.histplot(df['price_log'],kde=True)
plt.title('Target Distribution - prices')
# plt.show()

plt.subplot(3,3,5)
sns.boxplot(x=df['price_log'])
plt.title('Box Plot - prices')


plt.subplot(3,3,3)
sns.scatterplot(x=df['area'],y=df['price'])

plt.subplot(3,3,6)
sns.scatterplot(x=df['area'],y=df['price_log'])

plt.subplot(3,3,7)
sns.countplot(x=df['basement'])
plt.show()