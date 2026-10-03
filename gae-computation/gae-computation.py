def gae(rewards: list, values: list, gamma: float, lam: float) -> list:
    """
    Returns the generalized advantage estimate at every timestep.
    """
    T = len(rewards)
    adv = [0.0] * T
    last = 0.0
    for t in range(T, 0, -1):
        delta = rewards[t-1] + gamma * values[t] - values[t-1]
        adv[t-1] = delta + gamma * lam * last
        last = adv[t-1]
    return adv
        