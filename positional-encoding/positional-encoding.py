import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    res = np.zeros((seq_len, d_model), dtype=float)
    for pos in range(seq_len):
        for i in range(d_model):
            p = pos / (base**(2 * (i // 2) / d_model))
            res[pos][i] = np.sin(p) if i % 2 == 0 else np.cos(p)

    print(res)
    return res