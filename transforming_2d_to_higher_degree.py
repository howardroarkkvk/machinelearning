import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.preprocessing import PolynomialFeatures

# Generate a synthetic 2D dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=0.5,
    random_state=42
)

# Apply quadratic transformation (adds x1^2, x2^2, x1*x2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_quad = poly.fit_transform(X)
print(X_quad)
# For simplicity, we will plot: x1, x2, x1^2 + x2^2
Z = X_quad[:, 2] + X_quad[:, 4]  # x1^2 + x2^2

# 🔷 Plot original 2D data
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
ax1.set_title("Original 2D Space")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")

# 🔷 Plot transformed 3D data
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(X[:, 0], X[:, 1], Z, c=y, cmap='bwr', edgecolor='k')
ax2.set_title("Quadratic Transformed Feature Space")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_zlabel("x1² + x2²")

plt.tight_layout()
plt.show()
