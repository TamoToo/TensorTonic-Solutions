import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    n = len(X)
    w = np.zeros_like(X[0])
    b = 0.0
    for i in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        loss = - (1 / n) * np.sum([y[j] * np.log(p[j]) + (1 - y[j]) * np.log(1-p[j]) for j in range(n)])
        grad_w = (1 / n) * np.transpose(X) @ (p - y)
        grad_b = (1 / n) * np.sum([p[j] - y[j] for j in range(n)])
        w -= lr * grad_w
        b -= lr * grad_b

    return (w, b)
    
        