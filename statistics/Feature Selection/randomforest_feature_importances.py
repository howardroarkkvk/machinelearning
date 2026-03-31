import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# 1. Create sample dataset
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    random_state=42
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]

# 2. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Get feature importances
importances = model.feature_importances_

# 4. Put into dataframe for readability
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(importance_df)