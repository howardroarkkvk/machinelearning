import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(0)

class SimpleNN_NoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        print('x after relu',x)
        return self.fc2(x)

class SimpleNN_WithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        print('x after relu',x)
        return self.fc2(x)

# Random input
X = torch.randn(256, 100)

model_no_bn = SimpleNN_NoBN()
model_with_bn = SimpleNN_WithBN()

# Forward pass
with torch.no_grad():
    out_no_bn = model_no_bn(X)#.fc1(X)
    out_with_bn = model_with_bn(X)#.bn1(model_with_bn.fc1(X))

# Weights
weights_no_bn = model_no_bn.fc1.weight.view(-1).detach().numpy()
print(weights_no_bn)
weights_with_bn = model_with_bn.fc1.weight.view(-1).detach().numpy()
print(weights_with_bn)
# Plot
plt.figure(figsize=(14, 5))
sns.histplot(weights_no_bn, color="blue", bins=30, label="No BN", kde=True)
sns.histplot(weights_with_bn, color="red", bins=30, label="With BN", kde=True)
plt.legend()
plt.title("Weight Distribution With vs Without Batch Normalization")
plt.show()
