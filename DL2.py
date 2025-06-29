import numpy as np

def softmax(x):
    ex = np.exp(x - np.max(x, axis=1, keepdims=True))
    return ex / np.sum(ex, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Minimal training loop
X = np.array([[1, 2], [1.5, 2.5], [2, 3]])  # 3 samples, 2 features
y = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])  # One-hot encoded
W = np.random.randn(2, 3)  # Weights (2 features × 3 classes)
lr = 0.1

for epoch in range(100):
    # Forward pass
    logits = X @ W
    pred = softmax(logits)
    loss = cross_entropy(y, pred)
    
    # Backward pass
    grad = (pred - y) / len(X)
    dW = X.T @ grad
    
    # Update
    W -= lr * dW
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final predictions
print("\nFinal predictions:")
print(softmax(X @ W))
