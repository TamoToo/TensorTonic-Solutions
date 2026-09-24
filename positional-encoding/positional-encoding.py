import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    res = np.zeros((seq_len, d_model), dtype=float)
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angles = pos / (base ** (2 * (i // 2) / d_model))
    res[:, 0::2] = np.sin(angles[:, 0::2])  # Even indices: sin
    res[:, 1::2] = np.cos(angles[:, 1::2])
    return res