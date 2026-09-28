def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    # Write code here
    cum_ret = []
    w = 1.0
    for i in range(len(returns)):
        # print(w)
        w *= (1 + returns[i])
        cum_ret.append(w-1)
    return cum_ret