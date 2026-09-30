import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    return {
        "variance": float(np.var(x, ddof=1)),
        "standard_deviation": float(np.std(x, ddof=1)),
    }