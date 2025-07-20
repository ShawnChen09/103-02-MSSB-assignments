import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from simulator import simulator


def optimize_parameters(initial_k, steady_state, t):
    """
    Optimize the parameters for the mass-action model to match the steady state
    of the Michaelis-Menten model.

    Parameters:
        initial_k : Initial guess for the rate constants
        steady_state : Target steady state from the MM model
        t : Time points for the simulation

    Returns:
        Optimized rate constants
    """

    def objective(k):
        """
        Objective function for optimization.
        Minimizes the squared error between the steady state of the MA model
        and the target steady state from the MM model.
        """

        def ma_func(s):
            return np.array([s[i] * k[i] for i in range(3)])

        ss_model = odeint(simulator, y0=(0.3, 0.2, 0.1), t=t, args=(ma_func,))
        error = np.sum((ss_model[-1] - steady_state) ** 2)
        return error

    res = minimize(objective, initial_k, method="Nelder-Mead")

    return res.x, res.fun
