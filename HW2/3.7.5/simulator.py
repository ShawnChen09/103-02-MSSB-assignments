import numpy as np
from scipy.integrate import odeint


def simulator(s: float, _, vfunc):
    """
    The ODEs are:
        ds1/dt = 2 - v1 * s1
        ds2/dt = v1 * s1 - v2 * s2
        ds3/dt = v2 * s2 - v3 * s3

    Parameters:
        s : The concentration of the substrate at a given time.
        _ : Placeholder for time, not used in this equation.
        vfunc : The function used to calculate the velocities.

    Returns:
        A tuple representing the rate of change (ds1/dt, ds2/dt, ds3/dt).
    """
    v = vfunc(s)
    ds = [2 - v[0], v[0] - v[1], v[1] - v[2]]
    return ds


def run_simulation(t, vfunc, s00=(0.3, 0.2, 0.1), s10=(6, 4, 4)):
    """
    Run the simulation with two different initial conditions.

    Parameters:
        t : Time points for the simulation
        vfunc : The function used to calculate the velocities
        s00 : First initial condition (default: (0.3, 0.2, 0.1))
        s10 : Second initial condition (default: (6, 4, 4))

    Returns:
        s0, s1 : Simulation results for both initial conditions
    """
    s0 = odeint(simulator, y0=s00, t=t, args=(vfunc,))
    s1 = odeint(simulator, y0=s10, t=t, args=(vfunc,))

    return s0, s1
