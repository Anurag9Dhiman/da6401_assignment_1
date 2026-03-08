import numpy as np

class BaseOptimizer:
    def __init__(self, learning_rate=0.001, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def step(self, layers):
        raise NotImplementedError

class SGD(BaseOptimizer):
    def step(self, layers):
        for layer in layers:
            layer.W -= self.learning_rate * layer.grad_W
            layer.b -= self.learning_rate * layer.grad_b

class Momentum(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_W = None
        self.v_b = None

    def step(self, layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]
            
        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            
            layer.W -= self.learning_rate * self.v_W[i]
            layer.b -= self.learning_rate * self.v_b[i]

class NAG(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_W = None
        self.v_b = None
        self._W_orig = None
        self._b_orig = None

    def apply_lookahead(self, layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]
            
        self._W_orig = [np.copy(l.W) for l in layers]
        self._b_orig = [np.copy(l.b) for l in layers]
        
        for i, layer in enumerate(layers):
            layer.W -= self.beta * self.v_W[i]
            layer.b -= self.beta * self.v_b[i]

    def restore(self, layers):
        for i, layer in enumerate(layers):
            layer.W = self._W_orig[i]
            layer.b = self._b_orig[i]

    def step(self, layers):
        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            
            layer.W -= self.learning_rate * self.v_W[i]
            layer.b -= self.learning_rate * self.v_b[i]

class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.eg_W = None
        self.eg_b = None

    def step(self, layers):
        if self.eg_W is None:
            self.eg_W = [np.zeros_like(l.W) for l in layers]
            self.eg_b = [np.zeros_like(l.b) for l in layers]
            
        for i, layer in enumerate(layers):
            self.eg_W[i] = self.beta * self.eg_W[i] + (1 - self.beta) * (layer.grad_W**2)
            self.eg_b[i] = self.beta * self.eg_b[i] + (1 - self.beta) * (layer.grad_b**2)
            
            layer.W -= (self.learning_rate / (np.sqrt(self.eg_W[i]) + self.epsilon)) * layer.grad_W
            layer.b -= (self.learning_rate / (np.sqrt(self.eg_b[i]) + self.epsilon)) * layer.grad_b

def get_optimizer(name, learning_rate, weight_decay=0.0, **kwargs):
    name = name.lower()
    if name == 'sgd':
        return SGD(learning_rate, weight_decay)
    elif name == 'momentum':
        beta = kwargs.get('beta', 0.9)
        return Momentum(learning_rate, beta)
    elif name == 'nag':
        beta = kwargs.get('beta', 0.9)
        return NAG(learning_rate, beta)
    elif name == 'rmsprop':
        beta = kwargs.get('beta', 0.9)
        epsilon = kwargs.get('epsilon', 1e-8)
        return RMSProp(learning_rate, beta, epsilon)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
