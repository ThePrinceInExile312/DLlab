import numpy as np
import matplotlib.pyplot as plt

def get_x():
    return np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def lr(x):  # Leaky ReLU
    return np.where(x > 0, x, 0.01 * x)

def softmax(x):  # More stable version
    e_x = np.exp(x)
    return e_x / e_x.sum()

x = get_x()

plt.figure(figsize=(15, 10))

# Leaky ReLU
plt.subplot(2, 3, 1)
plt.plot(x, lr(x))
plt.title("Leaky ReLU")
plt.grid(True)

# Sigmoid
plt.subplot(2, 3, 2)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid")
plt.grid(True)

# ReLU
plt.subplot(2, 3, 3)
plt.plot(x, relu(x))
plt.title("ReLU")
plt.grid(True)

# Tanh
plt.subplot(2, 3, 4)
plt.plot(x, tanh(x))
plt.title("Tanh")
plt.grid(True)

# Softmax
plt.subplot(2, 3, 5)
plt.plot(x, softmax(x))
plt.title("Softmax")
plt.grid(True)

plt.show()
