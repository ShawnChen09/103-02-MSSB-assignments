from collections.abc import Callable
from typing import Tuple

import numpy as np


def my_solver(
    model: Callable,
    x0: float,
    tspan: tuple[float, float],
    dt: float,
    method: Callable,
) -> tuple[np.array, np.array]:
    """
    Solves an ODE using a specified numerical method.

    Parameters:
        model: The function representing the ODE.
        x0: The initial condition of x.
        tspan: A tuple representing the begin time and end time.
        dt: The time step for numerical integration.
        method: The numerical method to use for integration (e.g., euler, rk4).

    Returns:
        tuple:
        - ts: Array of time values.
        - xs: Array of computed x values at each time step.
    """
    t_begin, t_end = tspan
    ts = np.arange(t_begin, t_end, dt)
    xs = np.zeros(len(ts))
    xs[0] = x0

    for i, xs_i in enumerate(xs[:-1]):
        xs[i + 1] = method(model, xs_i, dt)

    return ts, xs
