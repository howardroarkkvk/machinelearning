from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_proba = model.predict_proba(X_test)[:, 1]
print(y_proba)
threshold = 0.5


while threshold >= 0:
    y_pred = (y_proba >= threshold).astype(int)
    print(f'y_pred is {y_pred}')
    f1 = f1_score(y_test, y_pred)

    print(f"Threshold: {threshold:.2f}, F1 Score: {f1:.4f}")

  
    threshold -= 0.02