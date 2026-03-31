import pandas as pd

df = pd.DataFrame({
    "gender": ["Male", "Female", "Male", "Female"]
})

# Before
print(df.memory_usage(deep=True))

# Convert
df["gender"] = df["gender"].astype("category")

# After
print(df.memory_usage(deep=True))