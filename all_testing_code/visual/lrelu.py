import numpy as np
import matplotlib.pyplot as plt

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * np.abs(x)

def double_lrelu(x, leak=0.1):
    return np.minimum(np.maximum(leak * x, x), leak * x - (leak - 1))

def leaky_clamp(x, lower, upper, leak=0.1):
    x_normalized = (x - lower) / (upper - lower)
    result = np.minimum(np.maximum(leak * x_normalized, x_normalized), leak * x_normalized - (leak - 1))
    return result * (upper - lower) + lower

x = np.linspace(-2, 2, 400)

# Compute the values for each function
y_lrelu = lrelu(x)
y_double_lrelu = double_lrelu(x)
y_leaky_clamp = leaky_clamp(x, lower=0, upper=1)

plt.figure(figsize=(12, 8))

# Plot lrelu
plt.subplot(3, 1, 1)
plt.plot(x, y_lrelu, label='lrelu')
plt.title('Leaky ReLU')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# Plot double_lrelu
plt.subplot(3, 1, 2)
plt.plot(x, y_double_lrelu, label='double_lrelu', color='orange')
plt.title('Double Leaky ReLU (Clamped to 0, 1)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# Plot leaky_clamp
plt.subplot(3, 1, 3)
plt.plot(x, y_leaky_clamp, label='leaky_clamp', color='green')
plt.title('Leaky ReLU with Clamping to [lower, upper]')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('leaky_relu_variants.png')
plt.close()
