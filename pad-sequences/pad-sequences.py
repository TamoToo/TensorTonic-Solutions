import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.array([], dtype=int).reshape(0, 0)
        
    if not max_len:
        max_len = len(max(seqs, key=len))

    results = []
    for seq in seqs:
        if len(seq) < max_len:
            results.append(seq + [pad_value] * (max_len - len(seq)))
        else:
            results.append(seq[:max_len])

    return np.asarray(results, dtype=int)