"""
Example script for running the eDNA model with predator-prey interactions.

This script demonstrates how to set up and run an eDNA model simulation
with predator-prey interactions.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import the edna_model package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edna_model.migration import combine_dvm, dvm
from edna_model.models import run_predator_simulation
from edna_model.physical_constants import (
    kappa_profile,
    vertical_advection_profile,
)
from edna_model.utils import setup_simulation_parameters
from edna_model.visualization import plot_concentration

# Set up basic simulation parameters
params = setup_simulation_parameters(days=4, z_max=1500.0, dz=0.5)
z = params["z"]
Nz = params["Nz"]
dz = params["dz"]
total_steps = params["total_steps"]
dt = params["dt"]
y0 = params["y0"]

# Settling rate (m/s), 25 m/d -> 0.002893.. m/ 10s
ws = -0.0 / 86400.0

# Vertical advection
wv = vertical_advection_profile(z)

# Decay rate (1/10s)
df = pd.read_csv("../decay_rate.csv")
k = df["Summer"] / 3600 * dt

# Diffusivity (m^2/10s)
kz = kappa_profile(z, season="summer", L_scale=2.0)
kz = 0  # Set to 0 as in the original code

# Breakdown rate (Large -> small)
delta_hour = 0.19  # 1/h
delta = delta_hour / 3600 * dt  # 1/10s

# Shedding rate: S_L + S_S = 1 / 10s
S_L = 0.5
S_S = 1.0 - S_L

# Create migration patterns
# Copepod (prey)
p1 = dvm(
    z,
    dz,
    int(86400 / dt),
    shallow_depth=50,
    deep_depth=200,
    ups_time=6,
    upe_time=9,
    downs_time=18,
    downe_time=21,
    layer_thickness=20,
)
p2 = dvm(
    z,
    dz,
    int(86400 / dt),
    shallow_depth=200,
    deep_depth=200,
    ups_time=18,
    upe_time=21,
    downs_time=6,
    downe_time=9,
    layer_thickness=20,
)
copepod_pattern = combine_dvm([p1, p2], [0.5, 0.5])

# Predator (e.g., fish that feeds on copepods)
predator_pattern = dvm(
    z,
    dz,
    int(86400 / dt),
    shallow_depth=40,
    deep_depth=55,
    ups_time=6,
    upe_time=9,
    downs_time=18,
    downe_time=21,
    layer_thickness=40,
)

# Predator parameters
digestion_time_steps = int(6.5 * 3600 / dt)  # 6.5 hours
release_duration_steps = min(int(1 * 3600 / dt), 100)  # Limit to 100 steps max
predation_efficiency = 0.9
release_decay_rate = 1.0

# Run the simulation
solution = run_predator_simulation(
    z,
    dz,
    total_steps,
    dt,
    copepod_pattern,
    predator_pattern,
    predation_efficiency,
    digestion_time_steps,
    release_decay_rate,
    release_duration_steps,
    wv,
    ws,
    k,
    delta,
    S_L,
    S_S,
    y0,
    method="euler",
)

C_L_sol = solution[:, :Nz].T
C_S_sol = solution[:, Nz:].T
sol = C_L_sol + C_S_sol

# Log transform for visualization
log10_sol = np.log10(sol)
log10_sol[log10_sol < -5] = -float("inf")

# Plot the results
fig = plot_concentration(
    z, 4, log10_sol, vmin=-1, vmax=2, depth_range=(0, 600)
)
# plt.savefig("predator_model_result.png", dpi=300, bbox_inches="tight")
plt.show()
