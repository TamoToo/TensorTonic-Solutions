import math

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    K = len(predictions)
    loss = 0.0
    for i in range(K):
        q = epsilon / K
        if i == target:
            q += 1 - epsilon
        loss -= q * math.log(predictions[i])
    return loss