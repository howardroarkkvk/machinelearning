import numpy as np
import matplotlib.pyplot as plt

# Define 2D contour of L1 constraint: |x| + |y| <= c (diamond shape)
c = 1.5
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_vals, y_vals)
L1_constraint = np.abs(X) + np.abs(Y)

# Define a base quadratic loss centered at (1, 1)
Loss = (X - 1)**2 + (Y - 1)**2

# Plot both views side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: 3D Loss Surface with L1 Ridge on (x + y)
Z = (X)**2 + (Y)**2 + 5 * np.abs(X + Y)
cs = axes[0].contourf(X, Y, Z, levels=50, cmap='plasma')
axes[0].plot(x_vals, -x_vals, 'w--', label='Ridge: x + y = 0')
axes[0].set_title('Loss Surface with Ridge (|x + y|)', fontsize=14)
axes[0].set_xlabel('Weight w1 (x)')
axes[0].set_ylabel('Weight w2 (y)')
axes[0].legend()

# --- Right: Constraint Region (Diamond) from |x| + |y| <= c
axes[1].contour(X, Y, L1_constraint, levels=[c], colors='red', linewidths=2)
loss_contours = axes[1].contour(X, Y, Loss, levels=20, cmap='gray')
axes[1].scatter(1, 1, color='blue', label='Unregularized Minimum (1,1)')
axes[1].set_title('L1 Constraint Region (|x| + |y| ≤ c)', fontsize=14)
axes[1].set_xlabel('Weight w1 (x)')
axes[1].set_ylabel('Weight w2 (y)')
axes[1].legend()

plt.tight_layout()
plt.show()
