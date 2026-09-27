def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    top_k = recommended[:k]
    intersection = len(set(top_k) & set(relevant))
    return [intersection / k, intersection / len(relevant)]