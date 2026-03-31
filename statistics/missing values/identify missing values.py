import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'age': [25, 30, None, 40, None],
    'salary': [50000, None, 60000, None, 70000],
    'city': ['A', 'B', None, 'A', 'C']
})

print(df)

# number of NA values in each column of dataframe
print(df.isnull().sum())

# percentages of NAs in each column when compared to the overall nulls i..e sum of all nulls in all columns
print(df.isnull().mean()*100)


# dropna it drops the rows which has nulls in any of the columns
# dropna(axis=1) it drops the columns which has nulls in any of the rows...
# df=df.dropna(axis=1)


# filling NA values with mean, median, mode etc 
print(df)
# df['age']=df['age'].fillna(df['age'].mean())
# df['age']=df['age'].fillna(0)
# df['salary']=df['salary'].fillna(df['salary'].median())
# df['city']=df['city'].fillna(df['city'].mode()[0])
# df['age']=df['age'].ffill()
# df=df.bfill()

# df['age']=df['age'].interpolate(method='linear')
df['age_missing'] = df['age'].isnull().astype(int)
print(df)
# sns.heatmap(df.isnull(),cbar=True)
# plt.show()
