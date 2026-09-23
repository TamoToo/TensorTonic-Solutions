def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    # Write code here
    n = len(series)
    mean = sum(series) / n
    variance = sum([(x - mean)**2 for x in series])
    result = [1.0]
    for i in range(1, max_lag + 1):
        if variance == 0:
            result.append(0.0)
            continue
        cov = (1 / variance) * sum([(series[k] - mean) * (series[k+i] - mean) for k in range(n - i)])
        result.append(round(cov, 6))

    return result
    