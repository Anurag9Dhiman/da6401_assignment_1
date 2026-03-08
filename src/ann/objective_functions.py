import numpy as np

def cross_entropy_loss(probs, y_one_hot, weight_decay=0.0, layers=None):
    """
    Cross-entropy loss: -1/N * sum(y * log(p))
    probs: (N, C)
    y_one_hot: (N, C)
    """
    N = probs.shape[0]
    # Small epsilon for stability
    probs = np.clip(probs, 1e-12, 1.0)
    loss = -np.sum(y_one_hot * np.log(probs)) / N
    
    if weight_decay > 0.0 and layers:
        reg_loss = 0.5 * weight_decay * sum(np.sum(l.W**2) for l in layers)
        loss += reg_loss
        
    return loss

def mse_loss(probs, y_one_hot, weight_decay=0.0, layers=None):
    """
    Mean Squared Error loss: 1/N * sum((p - y)^2)
    Note: The prompt mentions softmax+MSE gradient logic.
    """
    N = probs.shape[0]
    loss = np.mean(np.sum((probs - y_one_hot)**2, axis=1))
    
    if weight_decay > 0.0 and layers:
        reg_loss = 0.5 * weight_decay * sum(np.sum(l.W**2) for l in layers)
        loss += reg_loss
        
    return loss

def compute_loss(probs, y_one_hot, loss_name, weight_decay=0.0, layers=None):
    if loss_name == 'cross_entropy':
        return cross_entropy_loss(probs, y_one_hot, weight_decay, layers)
    elif loss_name == 'mse':
        return mse_loss(probs, y_one_hot, weight_decay, layers)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
