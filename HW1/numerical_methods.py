from collections.abc import Callable


def euler(model: Callable, x: float, dt: float) -> float:
    """
    Performs a single step of the Euler method for numerical integration.

    Parameters:
        model: The function representing the differential equation.
        x: The current value of x.
        dt: The time step for integration.

    Returns:
        The updated value of x after a single Euler step.
    """
    return x + dt * model(x)


def rk4(model: Callable, x: float, dt: float) -> float:
    """
    Performs a single step of the RK4 method for numerical integration.

    Parameters:
        model: The function representing the differential equation.
        x: The current value of x.
        dt: The time step for integration.

    Returns:
        The updated value of x after a single RK4 step.
    """
    k1 = dt * model(x)
    k2 = dt * model(x + 0.5 * k1)
    k3 = dt * model(x + 0.5 * k2)
    k4 = dt * model(x + k3)

    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
