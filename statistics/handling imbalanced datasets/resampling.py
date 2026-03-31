# Resampling adjusts the size of classes to make them more balanced.
# Oversampling: Duplicates or generates minority class samples to help the model learn more patterns.
# Undersampling: Removes majority class samples to balance the dataset and give equal importance to all classes.

import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# c=Counter('abccdddddabcdabcd')
# print(c)
# print(sorted(c))
# print(c.elements)

X,y=make_classification(n_classes=2,class_sep=2,weights=[0.1,0.9],
                    n_informative=3,n_redundant=1,flip_y=0,
                    n_features=20,n_clusters_per_class=1,
                    n_samples=1000,random_state=42)

# print(X)
print(Counter(y))

oversampler=RandomOverSampler(sampling_strategy='minority')
X_over,y_over=oversampler.fit_resample(X,y)

print(Counter(y_over),)

undersampler=RandomUnderSampler(sampling_strategy='majority')
X_under,y_under=undersampler.fit_resample(X,y)
print(Counter(y_under))