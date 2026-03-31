import pandas as pd

df = pd.DataFrame({
    "ID": [1, 2, 2, 3,3],
    "Age": [25, 30, 30, 35,35]
})

print(df)
# Find duplicates
print(df.duplicated())

# Show duplicate rows
print(df[df.duplicated()])

# Count duplicates
print(df.duplicated().sum())

# df=df.drop_duplicates(keep='last')
# print(df)

df=df.drop_duplicates(subset=['ID'])
print(df)