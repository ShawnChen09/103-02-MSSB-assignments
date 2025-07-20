import numpy as np


def gillespie_alg(model, u0, tend, params, stoich, tstart=0):
    """
    Gillespie algorithm for stochastic simulation.

    Parameters:
    -----------
    model : function
        Function that returns propensity values
    u0 : list or array
        Initial state
    tend : float
        End time for simulation
    params : dict
        Parameters for the model
    stoich : list
        Stoichiometry matrix
    tstart : float, optional
        Start time for simulation

    Returns:
    --------
    tuple
        (ts, us) - time points and corresponding states
    """
    t = tstart
    ts = [t]
    u0 = np.array(u0)
    u = u0.copy()
    us = [u.copy()]

    while t < tend:
        a = model(u, params)
        a_sum = sum(a)

        if a_sum <= 0:
            break

        # Only implement the "direct" method
        dt = np.random.exponential(1) / a_sum
        reaction = np.random.choice(len(stoich), p=np.array(a) / a_sum)
        u = u + stoich[reaction]
        t += dt

        ts.append(t)
        us.append(u.copy())

    return np.array(ts), np.array(us)
