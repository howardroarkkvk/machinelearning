import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create collinear features
x1 = np.linspace(-2, 2, 100)
x2 = 2 * x1**2  # perfectly collinear with x1
X = np.vstack([x1, x2]).T  # shape: (100, 2)
print('x is: ',X)
# I=np.ones(100)


# Step 2: Define the loss function
def loss_surface(w1_vals, w2_vals, X):
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    print(W1.shape,W2.shape)
    Z = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w = np.array([W1[i, j], W2[i, j]])
            # print('w shape is',w,w.shape)
            y_hat = (X @ w)**2
            # print('y hat is ',y_hat)
            # y_hat=y_hat+5*W1[]

            Z[i, j] = np.mean(y_hat)
            # print('z shape',Z[i,j],Z,Z.shape)

    return W1, W2, Z

# Step 3: Generate the loss surface
w_vals = np.linspace(-4, 4, 200)
W1, W2 = np.meshgrid(w_vals, w_vals)

W1, W2, Z = loss_surface(w_vals, w_vals, X)

# Step 4: Plot the contour
plt.figure(figsize=(6, 6))
contour = plt.contourf(W1, W2, Z, levels=50, cmap='coolwarm')
plt.colorbar(contour)
plt.xlabel("w1")
plt.ylabel("w2")
plt.title("Loss Surface: y = (wᵗx)² with x2 = 2·x1 (Collinear Features)")
plt.axis('equal')
plt.grid(True)
plt.show()