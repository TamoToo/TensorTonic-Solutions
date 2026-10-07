def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    """
    Returns the similarity-weighted rating prediction.
    """
    n = len(user_ratings)
    s = 0.0
    sr = 0.0
    for i in range(n):
        if i == target or item_similarities[i] <= 0.0 or user_ratings[i] <= 0.0:
            continue
        s += item_similarities[i]
        sr += item_similarities[i] * user_ratings[i]

    return float(sr / s) if s != 0.0 else 0.0