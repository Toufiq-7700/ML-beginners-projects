import numpy as np

# Dataset: [room, washroom, balcony, window]
# (Fake sample data for illustration)
X = np.array([
    [2, 1, 1, 4],
    [3, 2, 1, 6],
    [2, 1, 0, 3],
    [4, 2, 2, 8],
    [1, 1, 0, 2],
    [3, 1, 1, 5]
], dtype=float)

# Rent price (target variable)
y = np.array([15000, 25000, 12000, 32000, 10000, 22000], dtype=float)

# Hyperparameters
alpha = 0.00001     # learning rate
epochs = 50000      # iterations
m, n = X.shape      # m = samples, n = features
# print("Value of mn")
# print(m)
# print(n)
# Initialize parameters
w = np.zeros(n)     # one weight per feature
b = 0.0

# Gradient Descent
for epoch in range(epochs):
    y_pred = np.dot(X, w) + b   # predictions
    error = y_pred - y

    # Gradients
    dw = (2/m) * np.dot(X.T, error)   # vector of size (n,)
    db = (2/m) * np.sum(error)

    # Update
    w -= alpha * dw
    b -= alpha * db

# Final learned model
print(f"Final weights: {w}")
print(f"Final bias: {b}")

# Prediction for a new flat
x_new = np.array([3, 2, 1, 5])   # 3 rooms, 2 washrooms, 1 balcony, 5 windows
y_new = np.dot(x_new, w) + b
print(f"Predicted rent for {x_new} = {y_new:.2f} taka")
