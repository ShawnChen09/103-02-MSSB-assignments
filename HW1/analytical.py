import numpy as np


def analytical(t: float) -> float:
    """
    Compute the analytical solution of the ODE dx/dt = 1 - x.

    Parameters:
        t: The time value(s) at which to evaluate the solution.

    Returns:
        The analytical solution evaluated at time t.
    """
    return 1 - np.exp(-t)


def model(x: float, _=None) -> float:
    """
    Define the ODE dx/dt = 1 - x.

    Parameters:
        x : The value of x at a given time.
        _ (default to None): Placeholder for time, not used in this equation.

    Returns:
        The rate of change dx/dt.
    """
    return 1 - x
