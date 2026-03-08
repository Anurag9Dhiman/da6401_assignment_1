import numpy as np

def sigmoid(z):
    """
    Numerically stable sigmoid function.
    Clips z to [-500, 500] to avoid overflow/underflow.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    """
    Derivative of the sigmoid function: sigmoid(z) * (1 - sigmoid(z))
    Note: Usually, we use the cached activation for efficiency, 
    but the prompt asks for sigmoid_derivative(z).
    """
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh(z):
    """
    Hyperbolic tangent function.
    """
    return np.tanh(z)

def tanh_derivative(z):
    """
    Derivative of tanh: 1 - tanh(z)^2
    """
    t = np.tanh(z)
    return 1.0 - t**2

def relu(z):
    """
    Rectified Linear Unit function.
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    Derivative of ReLU: 1 if z > 0, else 0.
    """
    return (z > 0).astype(float)

def softmax(z):
    """
    Numerically stable softmax function.
    Subtracts the row-max for stability.
    Input z: (N, C)
    """
    # Subtract max per row for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

ACTIVATIONS = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu
}

ACTIVATION_DERIVATIVES = {
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'relu': relu_derivative
}

def get_activation(name):
    if name is None:
        return None
    return ACTIVATIONS.get(name.lower())

def get_activation_grad(name):
    if name is None:
        return None
    return ACTIVATION_DERIVATIVES.get(name.lower())
