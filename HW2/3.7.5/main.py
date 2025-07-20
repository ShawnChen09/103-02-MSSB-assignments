import numpy as np
from models import ma_model, mm_model
from optimization import optimize_parameters
from simulator import run_simulation

V_MAX = (9, 12, 15)
K_M = (1, 0.4, 3)
T = np.linspace(0, 5, 100)


def test_a():
    """Simulate the system from initial conditions (in mM) (s1, s2, s3) = (0.3, 0.2, 0.1).
    Repeat with initial condition (s1, s2, s3) = (6, 4, 4)."""

    def mme(s):
        return mm_model(s, V_MAX, K_M)

    mm_s0, mm_s1 = run_simulation(T, mme)
    return T, mm_s0, mm_s1


def test_b():
    """Optimize the parameters of a mass-action model to match the steady state
    of the Michaelis-Menten model."""
    _, mm_s0, _ = test_a()
    ss = mm_s0[-1]

    k_first = [V_MAX[i] / (K_M[i]) for i in range(3)]
    k_second = [V_MAX[i] / (K_M[i] + ss[i]) for i in range(3)]

    optimized_k, error = optimize_parameters(k_first, ss, T)
    print(f"Optimized parameters: {optimized_k}")
    print(f"Optimization error: {error}")

    return k_first, k_second, optimized_k


def test_c(k):
    """Run simulation with the Mass-action model."""

    def ma_func(s):
        return ma_model(s, k)

    ma_s0, ma_s1 = run_simulation(T, ma_func)

    return T, ma_s0, ma_s1
