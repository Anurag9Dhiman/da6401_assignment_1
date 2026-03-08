import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(name):
    """
    Loads raw dataset.
    Uses keras.datasets to avoid internet dependency (autograder compatibility).
    Returns: X_train, y_train, X_test, y_test as raw uint8.
    """
    print(f"Loading dataset {name}...")
    if name.lower() == 'mnist':
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif name.lower() == 'fashion_mnist':
        from tensorflow.keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {name}")
        
    # X shape: (N, 28, 28) uint8 - matches expected return format
    return X_train, y_train, X_test, y_test

def to_one_hot(y, num_classes=10):
    oh = np.zeros((y.size, num_classes))
    oh[np.arange(y.size), y] = 1
    return oh

def preprocess(X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, val_split=0.1, num_classes=10, seed=42):
    """
    1. Flatten to (N, 784)
    2. Normalize to [0, 1]
    3. Stratified split for validation
    4. One-hot encode targets
    """
    # Flatten and normalize
    X_tr = X_tr_raw.reshape(X_tr_raw.shape[0], -1).astype(float) / 255.0
    X_te = X_te_raw.reshape(X_te_raw.shape[0], -1).astype(float) / 255.0
    
    # Stratified split
    X_tr, X_val, y_tr_int, y_val_int = train_test_split(
        X_tr, y_tr_raw, test_size=val_split, stratify=y_tr_raw, random_state=seed
    )
    
    y_te_int = y_te_raw
    
    # One-hot encoding
    y_tr_oh = to_one_hot(y_tr_int, num_classes)
    y_val_oh = to_one_hot(y_val_int, num_classes)
    y_te_oh = to_one_hot(y_te_int, num_classes)
    
    return (X_tr, y_tr_oh, X_val, y_val_oh, X_te, y_te_oh, 
            y_tr_int, y_val_int, y_te_int)

def get_batches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
        
    for i in range(0, N, batch_size):
        batch_idx = indices[i : i + batch_size]
        yield X[batch_idx], y[batch_idx]

def get_samples_for_logging(X_2d, y, num_per_class=5):
    """
    X_2d: raw images (N, 28, 28)
    y: int labels
    """
    samples = []
    for cls in range(10):
        cls_indices = np.where(y == cls)[0]
        selected = np.random.choice(cls_indices, num_per_class, replace=False)
        for idx in selected:
            samples.append((X_2d[idx], cls))
    return samples
