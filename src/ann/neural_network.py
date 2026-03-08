import numpy as np
from .activations import get_activation, get_activation_grad, softmax
from .neural_layer import NeuralLayer
from .objective_functions import compute_loss

class NeuralNetwork:
    def __init__(self, *args, **kwargs):
        """
        Supports two positional calling conventions:
        Convention A: NeuralNetwork(input_size, hidden_sizes, num_classes)
        Convention B: NeuralNetwork(num_layers, hidden_sizes, num_classes) (num_layers < 10)
        Also supports keywords: input_size, hidden_sizes, num_classes, activation, weight_init
        """
        input_size = kwargs.get('input_size', 784)
        hidden_sizes = kwargs.get('hidden_sizes', [128, 128])
        num_classes = kwargs.get('num_classes', 10)
        activation = kwargs.get('activation', 'relu')
        weight_init = kwargs.get('weight_init', 'xavier')
        
        if len(args) >= 3:
            arg1, arg2, arg3 = args[:3]
            if isinstance(arg1, int) and arg1 < 10:
                # Convention B
                # Input size defaults to 784
                input_size = 784
                hidden_sizes = arg2
                num_classes = arg3
            else:
                # Convention A
                input_size = arg1
                hidden_sizes = arg2
                num_classes = arg3
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.activation = activation
        self.weight_init = weight_init
        
        self.layers = []
        
        # Build hidden layers
        curr_in = input_size
        for h_size in hidden_sizes:
            self.layers.append(NeuralLayer(curr_in, h_size, activation, weight_init))
            curr_in = h_size
            
        # Output layer (linear) - activation None as we want logits
        self.layers.append(NeuralLayer(curr_in, num_classes, activation=None, weight_init=weight_init))
        
        self._probs = None

    def __getattr__(self, name):
        """
        Proxy for Wi, bi, dWi, dbi.
        W1, b1, dW1, db1 ...
        1-indexed.
        """
        if name.startswith('W') and name[1:].isdigit():
            idx = int(name[1:]) - 1
            if 0 <= idx < len(self.layers):
                return self.layers[idx].W
        if name.startswith('b') and name[1:].isdigit():
            idx = int(name[1:]) - 1
            if 0 <= idx < len(self.layers):
                return self.layers[idx].b
        if name.startswith('dW') and name[2:].isdigit():
            idx = int(name[2:]) - 1
            if 0 <= idx < len(self.layers):
                return self.layers[idx].grad_W
        if name.startswith('db') and name[2:].isdigit():
            idx = int(name[2:]) - 1
            if 0 <= idx < len(self.layers):
                return self.layers[idx].grad_b
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """
        Pushes Wi, bi values into layer.W, layer.b.
        """
        if (name.startswith('W') or name.startswith('b')) and name[1:].isdigit():
            idx = int(name[1:]) - 1
            if hasattr(self, 'layers') and 0 <= idx < len(self.layers):
                if name.startswith('W'):
                    self.layers[idx].W = value
                else:
                    self.layers[idx].b = value
                return
        super().__setattr__(name, value)

    def forward(self, X):
        """
        Forward pass returning logits.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        curr_a = X
        for layer in self.layers:
            curr_a = layer.forward(curr_a)
        
        # Pre-activation of the last layer is stored in _Z, which is curr_a (logits)
        logits = curr_a
        self._probs = softmax(logits)
        return logits

    def backward(self, X, y, loss='cross_entropy', weight_decay=0.0):
        """
        Backward pass.
        Returns (dW_list, db_list) from last-layer to first-layer.
        Always recomputes forward pass to ensure _probs are fresh for gradient calculation.
        """
        self.forward(X)  # always recompute — never use stale _probs
            
        if loss == 'cross_entropy':
            delta = self._probs - y
        elif loss in ('mse', 'mean_squared_error'):
            diff = self._probs - y
            dot_diff_probs = np.sum(diff * self._probs, axis=1, keepdims=True)
            delta = 2.0 * self._probs * (diff - dot_diff_probs)
        else:
            raise ValueError(f"Unknown loss: {loss}")
            
        dw_list = []
        db_list = []
        
        curr_delta = delta
        for layer in reversed(self.layers):
            curr_delta = layer.backward(curr_delta, weight_decay)
            dw_list.append(layer.grad_W)
            db_list.append(layer.grad_b)
            
        return dw_list, db_list

    def get_weights(self):
        """
        Returns dict with W1, b1, ... and _config.
        Uses .copy() to ensure best_weights are not overwritten by reference.
        """
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            weights_dict[f'W{i+1}'] = layer.W.copy()
            weights_dict[f'b{i+1}'] = layer.b.copy()
            
        weights_dict['_config'] = {
            'input_size': self.input_size,
            'hidden_sizes': list(self.hidden_sizes),
            'num_classes': self.num_classes,
            'activation': self.activation,
            'weight_init': self.weight_init
        }
        return weights_dict

    def set_weights(self, weights_dict):
        """
        Restores weights from dict. 
        Infers architecture from weight shapes to avoid KeyError/Mismatches.
        """
        # Find all W keys in order
        w_keys = sorted(
            [k for k in weights_dict if k.startswith('W') and k[1:].isdigit()],
            key=lambda k: int(k[1:])
        )
        if not w_keys:
            return

        ws = [weights_dict[k] for k in w_keys]

        # Rebuild layers from weight shapes (fixes KeyError when counts mismatch)
        self.input_size = ws[0].shape[0]
        self.hidden_sizes = [w.shape[1] for w in ws[:-1]]
        self.num_classes = ws[-1].shape[1]
        
        if '_config' in weights_dict:
            cfg = weights_dict['_config']
            self.activation = cfg.get('activation', self.activation)
            self.weight_init = cfg.get('weight_init', self.weight_init)

        self.layers = []
        curr = self.input_size
        for h in self.hidden_sizes:
            self.layers.append(NeuralLayer(curr, h, self.activation, self.weight_init))
            curr = h
        # Output layer
        self.layers.append(NeuralLayer(curr, self.num_classes, None, self.weight_init))

        # Assign weights as copies
        for i, k in enumerate(w_keys):
            b_key = f'b{k[1:]}'
            self.layers[i].W = np.array(weights_dict[k]).copy()
            if b_key in weights_dict:
                self.layers[i].b = np.array(weights_dict[b_key]).copy()

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def get_gradient_norms(self):
        norms = {}
        for i, layer in enumerate(self.layers):
            norms[f'layer_{i+1}'] = layer.get_grad_norm()
        return norms

    def get_neuron_gradients(self, layer_idx, neuron_indices):
        """
        Returns gradient norms for specific neurons in a layer.
        grad_W is (in, out). Neuron idx refers to 'out' dimension.
        """
        if layer_idx >= len(self.layers):
            return np.array([])
        layer = self.layers[layer_idx]
        grad_w = layer.grad_W
        max_idx = grad_w.shape[1]
        safe_idxs = [int(i) for i in neuron_indices if int(i) < max_idx]
        return np.array([np.linalg.norm(grad_w[:, j]) for j in safe_idxs])
