import numpy as np
from .activations import get_activation, get_activation_grad

class NeuralLayer:
    def __init__(self, in_features, out_features, activation='relu', weight_init='xavier'):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.activation_func = get_activation(activation)
        self.activation_grad_func = get_activation_grad(activation)
        
        # Initialize weights and biases
        if weight_init == 'xavier':
            scale = np.sqrt(2.0 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * scale
        elif weight_init == 'random':
            self.W = np.random.randn(in_features, out_features) * 0.01
        elif weight_init == 'zeros':
            self.W = np.zeros((in_features, out_features))
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}")
            
        self.b = np.zeros((1, out_features))
        
        # Gradients
        self.grad_W = None
        self.grad_b = None
        
        # Caches
        self._X = None
        self._Z = None
        self._A = None

    def forward(self, X):
        """
        Forward pass.
        X shape: (N, in_features)
        """
        self._X = X
        self._Z = np.dot(X, self.W) + self.b
        
        if self.activation_func:
            self._A = self.activation_func(self._Z)
        else:
            # Output layer might not have an activation assigned via this class
            # (or it's the linear pre-activation the prompt wants)
            self._A = self._Z
            
        return self._A

    def backward(self, delta, weight_decay=0.0):
        """
        Backward pass.
        delta: Graduate of loss w.r.t. _A (for hidden) or _Z (for output)
        Note: The convention here is important. 
        If it's a hidden layer, delta passed in is dL/dA.
        We then compute dL/dZ = dL/dA * f'(Z).
        If it's the output layer, delta passed is usually dL/dZ (logits delta).
        """
        # If hidden layer with activation
        if self.activation_grad_func:
            dz = delta * self.activation_grad_func(self._Z)
        else:
            # Output layer (linear)
            dz = delta
            
        N = self._X.shape[0]
        
        self.grad_W = np.dot(self._X.T, dz) / N
        if weight_decay > 0.0:
            self.grad_W += weight_decay * self.W
            
        self.grad_b = np.sum(dz, axis=0, keepdims=True) / N
        
        # Return dX for the previous layer
        dX = np.dot(dz, self.W.T)
        return dX

    def get_grad_norm(self):
        if self.grad_W is None: return 0.0
        return np.linalg.norm(self.grad_W)

    def get_params(self):
        return self.W, self.b

    def set_params(self, W, b):
        self.W = W
        self.b = b
