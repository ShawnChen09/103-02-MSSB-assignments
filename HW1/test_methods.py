import numpy as np
from analytical import model
from numerical_methods import euler, rk4
from scipy.integrate import odeint
from solvers import my_solver


def test_odeint():
    t = np.linspace(0, 5, 100)
    x = odeint(model, y0=0, t=t)

    return t, x


def test_euler():
    methods = {}
    for dt in np.arange(0.1, 1.6, 0.1):
        tx = my_solver(model, x0=0, tspan=(0, 5), dt=dt, method=euler)
        methods[f"euler_{round(dt, 1)}"] = tx

    return methods


def test_rk4():
    methods = {}
    for dt in np.arange(0.1, 1.6, 0.1):
        tx = my_solver(model, x0=0, tspan=(0, 5), dt=dt, method=rk4)
        methods[f"rk4_{round(dt, 1)}"] = tx

    return methods
