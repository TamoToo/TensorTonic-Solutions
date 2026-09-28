def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    # Write code here
    cum_ret = []
    w = 1.0
    for r in returns:
        w *= (1 + r)
        cum_ret.append(w - 1)
    return cum_ret