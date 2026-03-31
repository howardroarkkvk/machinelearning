import pandas as pd
import os

path =r'E:\samples'
file_path=os.path.join(path,'Housing_2.csv')

df=pd.read_csv(file_path)
print(df)
print(df.info())

df_dtypes_int=df.select_dtypes(include=['int64'])
print(df_dtypes_int)

corr_df=df_dtypes_int.corr()['price'].abs().sort_values(ascending=False)
print(corr_df)

selected_features=corr_df[corr_df>0.4].index
print(selected_features)