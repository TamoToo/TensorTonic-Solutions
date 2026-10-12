def popularity_ranking(items: list, min_votes: int, global_mean: float) -> list:
    """
    Returns the weighted rating for every item.
    """
    return [
        (v * R) / (v + min_votes) + (min_votes * global_mean) / (v + min_votes)
        for R, v in items
    ]