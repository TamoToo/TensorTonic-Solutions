def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    # Write code here
    new_values = []
    for state in range(len(values)):
        action_values = []
        for action in range(len(transitions[state])):
            s = sum(transition * next_value for (transition, next_value) in zip(transitions[state][action], values))
            action_values.append(rewards[state][action] + gamma * s)

        new_values.append(max(action_values))

    return new_values