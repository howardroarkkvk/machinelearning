import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# print(type(sns.load_dataset('titanic').query("`class`=='First'")))

data1 = sns.load_dataset('titanic').query("`class` == 'First'")['age'].dropna()
data2 = sns.load_dataset('titanic').query("`class` == 'Third'")['age'].dropna()
print(data1.info())
print(data2.info())

data1=data1.to_frame()
data2=data2.to_frame()

data1['age_log']=np.log(data1)
data2['age_log']=np.log(data2)
print(data1.info())
print(data2.info())

data1_age_skew=data1.skew()

data2_age_skew=data2.skew()


print(data1_age_skew)

print(data2_age_skew)


print(data1.describe())
print(data2.describe())

print(data1['age'].mean(),data1['age'].median(),data1['age'].mode())
print(data1['age_log'].mean(),data1['age_log'].median(),data1['age_log'].mode())
print(data2['age'].mean(),data2['age'].median(),data2['age'].mode())
print(data2['age_log'].mean(),data2['age_log'].median(),data2['age_log'].mode())


# fig,axs=plt.subplots(2,2,figsize=(10,8))


# sns.histplot(data=data1['age'],label='First Class',color='blue',kde=True,ax=axs[0,0])
# sns.histplot(data=data1['age_log'],label='First log Class',color='blue',kde=True,ax=axs[1,0])
# sns.histplot(data=data2['age'],label='Third Class',color='red',kde=True,ax=axs[0,1])
# sns.histplot(data=data2['age_log'],label='Third log Class',color='red',kde=True,ax=axs[1,1])
# plt.tight_layout()
# plt.show()