import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import accuracy_score,classification_report
from collections import Counter

# this creates imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    weights=[0.1, 0.9],
    n_informative=3,
    n_redundant=1,
    random_state=42
)

print(Counter(y))

plt.figure()
plt.hist(y)
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.title("Imbalanced Class Distribution")
plt.show()

# splitting the dataset in to train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base_classifier=RandomForestClassifier(random_state=42)

balanced_bagging=BalancedBaggingClassifier(estimator=base_classifier,
                          sampling_strategy='auto',
                          replacement=False,
                          random_state=42)


balanced_bagging.fit(X_train,y_train)
y_pred=balanced_bagging.predict(X_test)

print(f'Accuracy ',accuracy_score(y_test,y_pred))
print(f'Classification report',classification_report(y_test,y_pred))