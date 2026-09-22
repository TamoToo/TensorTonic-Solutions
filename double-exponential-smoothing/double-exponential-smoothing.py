def double_exponential_smoothing(series: list, alpha: float, beta: float) -> list:
    """
    Returns the smoothed level at every time step.
    """
    # Write code here
    level = series[0]
    trend = series[1] - series[0]
    out = [level]
    for i in range(1, len(series)):
        new_level = alpha * series[i] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
        trend = new_trend
        out.append(level)
        
    return out