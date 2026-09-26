import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    # Write code here
    res = np.asarray(X, dtype=float).copy()
    nans = np.argwhere(np.isnan(X))
    if res.ndim == 1:
        stat = np.nanmean(X) if strategy == "mean" else np.nanmedian(X)
        if np.isnan(stat):
            stat = 0.0
        res[nans] = stat
        return res
        
    rows, cols = nans[:, 0], nans[:, 1]
    stat = np.nanmean(X, axis=0) if strategy == "mean" else np.nanmedian(X, axis=0)
    stat[np.isnan(stat)] = 0.0
    res[rows, cols] = stat[cols]
    return res
            