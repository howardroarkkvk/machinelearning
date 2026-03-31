import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing


data = fetch_california_housing()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target)

X=sm.add_constant(X)

def backward_elmination(X,y,significance_level=0.05):

    features=X.columns.tolist()
    while True:
        model=sm.OLS(y,X[features]).fit()
        p_values=model.pvalues
        max_p_value=p_values.max()

        if max_p_value>significance_level:
            features_to_remove=p_values.idxmax()
            print(f'removing:{features_to_remove} p-value:{max_p_value:.4f}')
            features.remove(features_to_remove)
        else:
            break
    return features,model

selected_features,final_model=backward_elmination(X,y)

print(f'selected features are :{selected_features}')
print(final_model.summary())