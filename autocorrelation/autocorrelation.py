def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    # Write code here
    mean = sum(series) / len(series)
    variance = sum([(x - mean)**2 for x in series])
    result = [1.0]
    for i in range(1, max_lag + 1):
        if variance == 0:
            result.append(0.0)
            continue
        auto_cor = (1 / variance) * sum([(series[k] - mean) * (series[k+i] - mean) for k in range(len(series) - i)])
        result.append(round(auto_cor, 6))

    return result
    