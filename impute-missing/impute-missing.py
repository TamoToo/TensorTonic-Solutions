import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    # Write code here
    res = np.asarray(X, dtype=float).copy()
    nans = np.argwhere(np.isnan(X))
    stats = np.nanmean(X, axis=0) if strategy == "mean" else np.nanmedian(X, axis=0)
    if res.ndim == 1:
        if np.isnan(stats):
            stats = 0.0
        res[nans] = stats
        return res
        
    rows, cols = nans[:, 0], nans[:, 1]
    stats[np.isnan(stats)] = 0.0
    res[rows, cols] = stats[cols]
    return res
            