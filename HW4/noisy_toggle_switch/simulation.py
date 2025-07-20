import numpy as np
from gillespie import gillespie_alg
from model import model
from tqdm import tqdm


def run_simulations(alpha_values, u0, tend=10, num_runs=1):
    """
    Run multiple simulations for different alpha values.

    Args:
        alpha_values (list): List of alpha values to simulate.
        u0 (list): Initial state [N1, N2].
        tend (float, optional): End time for simulation. Defaults to 10.
        num_runs (int, optional): Number of runs for each alpha value. Defaults to 1.

    Returns:
        dict: Results for each alpha value.
    """
    results = {}

    delta = 1.0
    beta = 4.0

    stoich = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
    ]

    for alpha in alpha_values:
        params = {"alpha": alpha, "beta": beta, "delta": delta}
        alpha_results = []

        for _ in tqdm(range(num_runs), desc=f"Alpha = {alpha}"):
            ts, us = gillespie_alg(model, u0, tend, params, stoich)
            alpha_results.append((ts, us))

        results[alpha] = alpha_results

    return results
