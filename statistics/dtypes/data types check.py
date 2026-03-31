import pandas as pd

df = pd.DataFrame({
    "age": ["25", "30", "35"],
    "salary": ["50000", "60000", "not_available"],
        "date": ["2024-01-01", "2024-02-01","hi"],
            "is_active": ["yes", "no", "yes"]
})

print(df.info())
print('--------------')
print(df.dtypes)
print('--------------')
print(df.columns)
print(df)
print('--------------')

df['age']=df['age'].astype(int)
df['salary']=pd.to_numeric(df['salary'],errors='coerce')
df['date']=pd.to_datetime(df['date'],errors='coerce')
df['is_active']=df['is_active'].map({'yes':1,'no':0})


print(df.info())
print('--------------')
print(df.dtypes)
print('--------------')
print(df.columns)
print('--------------')
print(df)

print(df['date'].dt.year)
print(df['date'].dt.month)
print(df['salary'].unique())
print(df['salary'].nunique())