import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    res = np.zeros((seq_length, d_model))
    pos = np.arange(seq_length).reshape(-1, 1)
    i = np.arange(d_model).reshape(1, -1)
    angles = pos / (10000.0 ** (2 * (i // 2) / d_model))
    res[:, 0::2] = np.sin(angles[:, 0::2])
    res[:, 1::2] = np.cos(angles[:, 1::2])
    return res