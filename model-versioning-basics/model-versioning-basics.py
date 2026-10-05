def promote_model(models: list) -> str:
    """
    Returns the model name as a string.
    """
    best = models[0]
    for model in models[1:]:
        better_acc = model["accuracy"] > best["accuracy"]
        eq_acc = model["accuracy"] == best["accuracy"]
        better_latency = model["latency"] < best["latency"]
        eq_latency = model["latency"] == best["latency"]
        newer = model["timestamp"] > best["timestamp"]
        if better_acc or (eq_acc and better_latency) or (eq_acc and eq_latency and newer):
            best = model
    return best["name"]