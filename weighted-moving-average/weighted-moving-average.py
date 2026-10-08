def weighted_moving_average(values: list, weights: list) -> list:
    """
    Returns the weighted average of every complete window.
    """
    size = len(weights)
    w_sum = sum(weights)
    res = []
    for i in range(len(values) - size + 1):
        total = sum(weights[j] * values[i + j] for j in range(size))
        res.append(total / w_sum)
    return res