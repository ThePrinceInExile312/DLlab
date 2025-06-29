import numpy as np

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(float)

def softmax(z):
    z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return z_exp / np.sum(z_exp, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

# ---------- Dataset (3 samples, 2 features, 3 classes) ----------

X = np.array([
    [0.5, 1.5],
    [1.0, 2.0],
    [1.5, 2.5]
])

Y = np.eye(3)
hidden_units = 4      # Number of neurons in hidden layer
learning_rate = 0.1   # Step size for SGD updates
epochs = 600          # Number of training iterations

W1 = 0.01 * np.random.randn(X.shape[1], hidden_units)  # Weights input->hidden
b1 = np.zeros((1, hidden_units))                       # Biases hidden layer

W2 = 0.01 * np.random.randn(hidden_units, Y.shape[1]) # Weights hidden->output
b2 = np.zeros((1, Y.shape[1]))                         # Biases output layer

# ---------- Training loop ----------

for epoch in range(epochs):
    # Forward pass
    Z1 = X @ W1 + b1     # Linear transform hidden layer
    A1 = relu(Z1)        # Activation hidden layer
    Z2 = A1 @ W2 + b2    # Linear transform output layer
    P = softmax(Z2)      # Predicted probabilities

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        loss = cross_entropy(Y, P)
        print(f"Epoch {epoch:3d} - Loss: {loss:.4f}")

    # Backward pass (compute gradients)
    dZ2 = (P - Y) / len(X)       # Gradient output layer (softmax + cross-entropy)
    dW2 = A1.T @ dZ2             # Gradient weights hidden->output
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # Gradient biases output layer

    dA1 = dZ2 @ W2.T             # Gradient activations hidden layer
    dZ1 = dA1 * drelu(Z1)        # Gradient linear hidden layer
    dW1 = X.T @ dZ1              # Gradient weights input->hidden
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # Gradient biases hidden layer

    # SGD parameter update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

print("\nFinal probabilities:\n", P)
