import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    # Write code here
    x = np.asarray(x, dtype=int)
    p = np.asarray(p, dtype=float)
    return np.sum([x[i] * p[i] for i in range(len(x))])