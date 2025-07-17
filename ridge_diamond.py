import numpy as np
import matplotlib.pyplot as plt

# Define the base loss: centered at (1.5, 0.5)
def base_loss(x, y):
    return (x - 1.5)**2 + (y - 0.5)**2

# Penalty-based version (ridge in 3D)
def l1_penalty_loss(x, y, lambd=1.5):
    return base_loss(x, y) + lambd * (np.abs(x) + np.abs(y))

# Grid for plotting
x_vals = np.linspace(-2, 3, 300)
y_vals = np.linspace(-2, 3, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z_penalty = l1_penalty_loss(X, Y)

# Constraint-based version: find lowest base loss inside diamond
constraint_bound = 1.5  # equivalent to lambda=1.5
Z_base = base_loss(X, Y)
L1_norm = np.abs(X) + np.abs(Y)
Z_constrained = np.where(L1_norm <= constraint_bound, Z_base, np.nan)

# Find approximate minima
penalty_min = np.unravel_index(np.nanargmin(Z_penalty), Z_penalty.shape)
constrained_min = np.unravel_index(np.nanargmin(Z_constrained), Z_constrained.shape)

penalty_point = (X[penalty_min], Y[penalty_min])
constrained_point = (X[constrained_min], Y[constrained_min])

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Penalty-based loss (with ridge)
cp1 = axes[0].contourf(X, Y, Z_penalty, levels=50, cmap='plasma')
axes[0].plot(*penalty_point, 'ro', label='Penalty Min')
axes[0].set_title('Penalty-based (Ridge) Loss Surface')
axes[0].set_xlabel('Weight w1 (x)')
axes[0].set_ylabel('Weight w2 (y)')
axes[0].legend()
fig.colorbar(cp1, ax=axes[0])

# Right: Constrained version (inside diamond)
cp2 = axes[1].contourf(X, Y, Z_base, levels=50, cmap='viridis')
axes[1].contour(X, Y, L1_norm, levels=[constraint_bound], colors='red', linewidths=2, linestyles='--')
axes[1].plot(*constrained_point, 'bo', label='Constrained Min')
axes[1].set_title('Constraint-based (Diamond) Optimization')
axes[1].set_xlabel('Weight w1 (x)')
axes[1].set_ylabel('Weight w2 (y)')
axes[1].legend()
fig.colorbar(cp2, ax=axes[1])

plt.tight_layout()
plt.show()
