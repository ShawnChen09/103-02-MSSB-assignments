import numpy as np


def model(u, params):
    """
    Define the model for the noisy toggle switch.

    Args:
        u (list or array): Current state [N1, N2].
        params (dict): Parameters for the model including alpha, beta, delta.

    Returns:
        list: Propensity values [a1, a2, a3, a4].
    """
    N1, N2 = u
    alpha, beta, delta = params["alpha"], params["beta"], params["delta"]
    a1 = alpha / (1 + N2**beta)
    a2 = alpha / (1 + N1**beta)
    a3 = delta * N1
    a4 = delta * N2
    return [a1, a2, a3, a4]
