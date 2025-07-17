import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Let's simulate gradient descent from a starting point and show the path toward the ridge

x=np.linspace(-4,4) # by default linspace creates 100 intervals it inlueds the boundaries...
y=np.linspace(-4,4)


X,Y=np.meshgrid(x,y)


# Define the loss gradient manually
def gradient(x, y, lambda_val=5.0):
    grad_x = 2 * x + lambda_val * np.sign(x + y) #this return +1 for x+y>1 and 0 for x+y=0 and -1 for x+y<0
    grad_y = 2 * y + lambda_val * np.sign(x + y)
    return grad_x, grad_y

# Perform gradient descent
x_path = [2.0]
y_path = [-1.0]
learning_rate = 0.05
steps = 5

x_curr, y_curr = x_path[0], y_path[0]

for _ in range(steps):
    grad_x, grad_y = gradient(x_curr, y_curr)
    x_curr -= learning_rate * grad_x
    y_curr -= learning_rate * grad_y
    x_path.append(x_curr)
    y_path.append(y_curr)

# Compute the contour surface
Z2 = 1 + X**2 + Y**2 + 5 * np.abs(X + Y)

# Plotting the contour with gradient descent path
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z2, levels=50, cmap='plasma')
cbar = plt.colorbar(contour)

# Ridge line x + y = 0
ax.plot(x, -x, color='white', linestyle='--', linewidth=2, label='x + y = 0 (ridge)')

# Plot gradient descent path
ax.plot(x_path, y_path, color='cyan', marker='o', markersize=3, label='Gradient Descent Path')
ax.scatter(x_path[0], y_path[0], color='lime', s=60, label='Start Point')
ax.scatter(x_path[-1], y_path[-1], color='red', s=60, label='End Point')

# Labels
ax.set_xlabel('Weight w1 (x)', fontsize=12)
ax.set_ylabel('Weight w2 (y)', fontsize=12)
ax.set_title('Gradient Descent Toward Ridge (x + y = 0)', fontsize=14)
ax.legend()

plt.tight_layout()
plt.show()
