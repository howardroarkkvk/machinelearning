import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Sample data
df = pd.DataFrame({
    'A': [1, 1, 1, 1, 1],
    'B': [0, 1, 0, 1, 0],
    'C': [10, 20, 30, 40, 50]
})

print(df.var())
# Apply Variance Threshold
selector = VarianceThreshold(threshold=0.3)
X_selected = selector.fit_transform(df)

# Get selected feature names
print(selector.get_feature_names_out())
print(selector.get_params())
print(selector.variances_)
selected_features = df.columns[selector.get_support()]

print(selected_features)



variance_df = pd.DataFrame({
    'Feature': df.columns,
    'Variance': selector.variances_,
    'Selected': selector.get_support()
})

print(variance_df)