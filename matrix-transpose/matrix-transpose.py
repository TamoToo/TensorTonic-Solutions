import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    At = np.zeros((m, n), dtype=float)
    for i in range(n):
        for j in range(m):
            At[j][i] = A[i][j]
    return At
