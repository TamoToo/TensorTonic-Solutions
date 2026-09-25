import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    # mean
    mu, sigma = np.mean(x, axis=-1, keepdims=True), np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(sigma + eps)
    return gamma * x_norm + beta