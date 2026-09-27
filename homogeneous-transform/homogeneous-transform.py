import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    # Write code here
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)
    single_point = points.ndim == 1
    if single_point:
        points = np.pad(points, (0, 1), constant_values=1)
    else:
        points = np.pad(points, pad_width=((0, 0), (0, 1)), constant_values=1)
    new_points = (T @ points.T).T
    return new_points[:, :3] if not single_point else new_points[:3]