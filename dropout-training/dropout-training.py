import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    probs = rng.random(x.shape)
    mask = (probs < (1.0 - p)).astype(float) / (1.0 - p)
    return (x * mask, mask)
    