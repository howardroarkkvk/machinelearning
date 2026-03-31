from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

data = pd.DataFrame({
    'Experience': [1, 2, 3, 4, 5, 6],
    'Age': [22, 25, 30, 35, 40, 45],
    'Salary': [20000, 25000, 30000, 35000, 40000, 45000],
    'Purchased': [0, 0, 0, 1, 1, 1]
})

X = data[['Experience', 'Age', 'Salary']]
y = data['Purchased']

remaining_features=list(X.columns)
print(f'remaining features are {remaining_features}')
features_selected=[]
best_score=0

while remaining_features:
    scores=[]
    for feature in remaining_features:
        temp_features=features_selected+[feature]
        print(temp_features)
        model=LogisticRegression()
        score=cross_val_score(model,data[temp_features],y,cv=3).mean()
        print(f'score in for loop for feature: {feature} ',score)
        scores.append((score,feature))
    print(f'{scores}')
    scores.sort(reverse=True)
    best_new_score,best_feature=scores[0]

    if best_new_score<=best_score:
        break

    features_selected.append(best_feature)
    remaining_features.remove(best_feature)
    best_score=best_new_score

    print(f'Added : {best_feature},score:{best_new_score}')


print(f'final selected features',features_selected)