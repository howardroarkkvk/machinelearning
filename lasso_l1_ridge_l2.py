import matplotlib.pyplot as plt
import numpy as np

# Create a grid of w1 and w2 values
w1 = np.linspace(-2, 2, 400)
w2 = np.linspace(-2, 2, 400)
W1, W2 = np.meshgrid(w1, w2)

# Simulated loss contours (e.g., squared error loss)
loss = (W1 - 1)**2 + (W2 - 1.5)**2  # Minimum at (1, 1.5)

# L1 constraint (diamond): |w1| + |w2| <= t
t = 1.5
l1_constraint = np.abs(W1) + np.abs(W2)

# L2 constraint (circle): w1^2 + w2^2 <= r^2
r = 1.3
l2_constraint = W1**2 + W2**2

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot L1 case
ax[0].contour(W1, W2, loss, levels=20, cmap='gray')
ax[0].contour(W1, W2, l1_constraint, levels=[t], colors='red', linewidths=2)
ax[0].plot(1, 1.5, 'go', label='Loss Minimum')
ax[0].set_title('L1 Regularization (Lasso)\nDiamond constraint → Sparse solution')
ax[0].set_xlabel('w1')
ax[0].set_ylabel('w2')
ax[0].legend()
ax[0].grid(True)

# Plot L2 case
ax[1].contour(W1, W2, loss, levels=20, cmap='gray')
ax[1].contour(W1, W2, l2_constraint, levels=[r**2], colors='blue', linewidths=2)
ax[1].plot(1, 1.5, 'go', label='Loss Minimum')
ax[1].set_title('L2 Regularization (Ridge)\nCircle constraint → Smooth shrinkage')
ax[1].set_xlabel('w1')
ax[1].set_ylabel('w2')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
