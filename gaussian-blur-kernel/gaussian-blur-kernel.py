import math

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    mat = []
    center = size // 2
    total = 0.0
    for i in range(size):
        row = []
        for j in range(size):
            x = i - center
            y = j - center
            g = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            total += g
            row.append(g)
        mat.append(row)

    return [[g / total for g in rows] for rows in mat]
        