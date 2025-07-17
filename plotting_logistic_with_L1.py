import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Create synthetic binary classification data with 2 informative features
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression with L1 regularization (Lasso)
clf_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)  # Smaller C = stronger regularization
clf_l1.fit(X_scaled, y)

# Train logistic regression without regularization (for comparison)
clf_none = LogisticRegression(penalty='l2', solver='lbfgs')
clf_none.fit(X_scaled, y)

# Extract weights
w_l1 = clf_l1.coef_[0]
w_none = clf_none.coef_[0]

# Plot weight vectors
fig, ax = plt.subplots(figsize=(8, 6))
ax.arrow(0, 0, w_none[0], w_none[1], head_width=0.05, color='blue', label='No Regularization', length_includes_head=True)
ax.arrow(0, 0, w_l1[0], w_l1[1], head_width=0.05, color='red', label='L1 Regularization', length_includes_head=True)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('Weight w1')
ax.set_ylabel('Weight w2')
ax.set_title('Effect of L1 Regularization on Logistic Regression Weights')
ax.grid(True)
ax.legend()
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.tight_layout()
plt.show()
