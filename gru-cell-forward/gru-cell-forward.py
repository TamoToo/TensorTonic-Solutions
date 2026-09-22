import numpy as np

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    # Write code here
    def sigmoid(z):
        z = np.asarray(z, dtype=float)
        return 1.0 / (1.0 + np.exp(-z))

    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    params = {name: np.asarray(value, dtype=float) for name, value in params.items()}
    single_sample = x.ndim == 1
    if single_sample:
        x = x.reshape(1, -1)
        h_prev = h_prev.reshape(1, -1)
        
    update_gate = sigmoid(x @ params["Wz"] + h_prev @ params["Uz"] + params["bz"])
    reset_gate = sigmoid(x @ params["Wr"] + h_prev @ params["Ur"] + params["br"])
    candidate_hidden_state = np.tanh(
        x @ params["Wh"] + (reset_gate * h_prev) @ params["Uh"] + params["bh"]
    )
    new_hidden_state = (1 - update_gate) * h_prev + update_gate * candidate_hidden_state
    return new_hidden_state[0] if single_sample else new_hidden_state