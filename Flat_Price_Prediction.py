import numpy as np
# Dataset 
x = np.array([
    700, 729, 759, 788, 818, 848, 877, 907, 936, 966,
    995, 1025, 1055, 1084, 1114, 1143, 1173, 1202, 1232, 1262,
    1291, 1321, 1350, 1380, 1409, 1439, 1469, 1498, 1528, 1557,
    1587, 1616, 1646, 1676, 1705, 1735, 1764, 1794, 1823, 1853,
    1883, 1912, 1942, 1971, 2000
], dtype=float)

y = np.array([
     9096,  8822, 10254, 11793, 10173, 11391, 11325, 13533, 13611, 13105,
    13768, 14447, 14801, 15465, 15321, 16335, 17188, 17400, 18183, 19109,
    19445, 20274, 21025, 21664, 21463, 22742, 22594, 23635, 23922, 24686,
    25044, 25677, 25984, 27131, 26572, 27594, 27897, 28837, 28729, 30003,
    29382, 30352, 30816, 31386, 31752
], dtype=float)

# Hyperparameters
alpha = 0.00000001   # learning rate
epochs = 50000       # more epochs for convergence

# Initialize parameters
w, b = 0.0, 0.0
m = len(x)

# Gradient Descent
for epoch in range(epochs):
    y_pred = w * x + b
    error = y_pred - y

    # Gradients
    dw = (2/m) * np.dot(error, x)
    db = (2/m) * np.sum(error)

    # Update
    w -= alpha * dw
    b -= alpha * db

# Final learned model
print(f"Final w (price per sqft): {w:.4f}")
print(f"Final b (base price): {b:.4f}")

# Prediction for 1250 sqft
x_new = 1750
y_new = w * x_new + b
print(f"Predicted cost for {x_new} sqft flat = {y_new:.2f} taka")
