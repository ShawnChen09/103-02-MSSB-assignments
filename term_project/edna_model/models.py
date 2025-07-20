"""
eDNA models for simulating concentration dynamics.

This module contains functions for simulating the dynamics of eDNA concentration
in the water column, including basic and predator-prey models.
"""

from typing import Literal

import numpy as np
from numba import jit
from tqdm import tqdm


def model(y_current, source_force, kz, wv, ws, k, delta, Nz):
    """
    Basic model for eDNA concentration dynamics.

    Parameters
    ----------
    y_current : array_like
        Current state vector containing large and small eDNA concentrations.
    source_force : array_like
        Source term for eDNA input.
    kz : array_like
        Diffusivity profile.
    wv : array_like
        Vertical advection profile.
    ws : float
        Settling rate.
    k : array_like
        Decay rate profile.
    delta : float
        Breakdown rate from large to small eDNA.
    Nz : int
        Number of depth points.

    Returns
    -------
    array_like
        Rate of change of eDNA concentrations.
    """
    C_L = y_current[:Nz]
    C_S = y_current[Nz:]

    dC_L_dz = np.gradient(C_L, 0.5)  # Assuming dz = 0.5
    dC_S_dz = np.gradient(C_S, 0.5)
    diffusion_L = np.gradient(kz * dC_L_dz, 0.5)
    diffusion_S = np.gradient(kz * dC_S_dz, 0.5)

    dC_L_dt = (
        -(wv - ws) * dC_L_dz
        + diffusion_L
        - (k + delta) * C_L
        + source_force * 0.5  # Assuming S_L = 0.5
    )

    dC_S_dt = (
        -wv * dC_S_dz
        + diffusion_S
        - k * C_S
        + delta * C_L
        + source_force * 0.5  # Assuming S_S = 0.5
    )

    return np.concatenate([dC_L_dt, dC_S_dt])


def run_simulation(
    z,
    dz,
    total_steps,
    source_mx,
    kz,
    wv,
    ws,
    k,
    delta,
    S_L,
    S_S,
    y0,
    method: Literal["euler", "rk4"] = "euler",
):
    """
    Run a simulation of eDNA concentration dynamics.

    Parameters
    ----------
    z : array_like
        Depth values (m).
    dz : float
        Depth resolution (m).
    total_steps : int
        Total number of time steps.
    source_mx : array_like
        Source matrix for eDNA input over time and depth.
    kz : array_like
        Diffusivity profile.
    wv : array_like
        Vertical advection profile.
    ws : float
        Settling rate.
    k : array_like
        Decay rate profile.
    delta : float
        Breakdown rate from large to small eDNA.
    S_L : float
        Shedding rate for large eDNA.
    S_S : float
        Shedding rate for small eDNA.
    y0 : array_like
        Initial state vector.
    method : str, optional
        Integration method. Options: "euler", "rk4".

    Returns
    -------
    array_like
        Array of eDNA concentrations over time and depth.
    """
    Nz = len(z)
    concentration = []
    y = y0.copy()

    for t in tqdm(range(total_steps), desc="Simulate concentration dynamic"):
        time_idx = int(t % len(source_mx))
        source_force = source_mx[time_idx]

        # RK4 method
        k1 = model(y, source_force, kz, wv, ws, k, delta, Nz)
        if method == "rk4":
            k2 = model(y + 0.5 * k1, source_force, kz, wv, ws, k, delta, Nz)
            k3 = model(y + 0.5 * k2, source_force, kz, wv, ws, k, delta, Nz)
            k4 = model(y + k3, source_force, kz, wv, ws, k, delta, Nz)

        y += k1 if method == "euler" else (k1 + 2 * k2 + 2 * k3 + k4) / 6

        concentration.append(y.copy())

    return np.array(concentration)  # shape: (Nt, 2*Nz)


@jit(nopython=True)
def calculate_predation_interaction(
    copepod_dist, predator_dist, C_L, C_S, predation_efficiency
):
    """
    Calculate predation interaction between predator and prey.

    Parameters
    ----------
    copepod_dist : array_like
        Distribution of copepods.
    predator_dist : array_like
        Distribution of predators.
    C_L : array_like
        Large eDNA concentration.
    C_S : array_like
        Small eDNA concentration.
    predation_efficiency : float
        Efficiency of predation.

    Returns
    -------
    tuple
        Consumed large and small eDNA.
    """
    overlap = copepod_dist * predator_dist
    consumed_L = predation_efficiency * overlap * C_L
    consumed_S = predation_efficiency * overlap * C_S
    return consumed_L, consumed_S


@jit(nopython=True)
def update_predator_stomach(
    consumed_L, consumed_S, stomach_idx, predator_stomach
):
    """
    Update predator stomach contents.

    Parameters
    ----------
    consumed_L : array_like
        Consumed large eDNA.
    consumed_S : array_like
        Consumed small eDNA.
    stomach_idx : int
        Current stomach index.
    predator_stomach : array_like
        Predator stomach contents.

    Returns
    -------
    array_like
        Updated predator stomach contents.
    """
    predator_stomach[stomach_idx, :, 0] = consumed_L
    predator_stomach[stomach_idx, :, 1] = consumed_S
    return predator_stomach


@jit(nopython=True)
def calculate_predator_release(
    predator_stomach,
    stomach_idx,
    predator_dist,
    digestion_time_steps,
    precomputed_release_factors,
):
    """
    Calculate eDNA release from predator.

    Parameters
    ----------
    predator_stomach : array_like
        Predator stomach contents.
    stomach_idx : int
        Current stomach index.
    predator_dist : array_like
        Distribution of predators.
    digestion_time_steps : int
        Number of time steps for digestion.
    precomputed_release_factors : array_like
        Precomputed release factors.

    Returns
    -------
    tuple
        Released large and small eDNA, and updated predator stomach contents.
    """
    total_released_L = np.zeros_like(predator_dist)
    total_released_S = np.zeros_like(predator_dist)

    stomach_len = len(predator_stomach)

    max_iterations = min(len(precomputed_release_factors), stomach_len)

    for i in range(max_iterations):
        release_idx = (stomach_idx - digestion_time_steps - i) % stomach_len

        if release_idx >= 0:
            stomach_L_sum = np.sum(predator_stomach[release_idx, :, 0])
            stomach_S_sum = np.sum(predator_stomach[release_idx, :, 1])

            if stomach_L_sum < 1e-15 and stomach_S_sum < 1e-15:
                continue

            release_fraction = precomputed_release_factors[i]

            released_factor = release_fraction
            total_released_L += released_factor * stomach_L_sum * predator_dist
            total_released_S += released_factor * stomach_S_sum * predator_dist

            decay_factor = 1.0 - release_fraction
            predator_stomach[release_idx, :, 0] *= decay_factor
            predator_stomach[release_idx, :, 1] *= decay_factor

    return total_released_L, total_released_S, predator_stomach


def compute_derivatives(C_L, C_S, dz):
    """
    Compute derivatives for eDNA concentrations.

    Parameters
    ----------
    C_L : array_like
        Large eDNA concentration.
    C_S : array_like
        Small eDNA concentration.
    dz : float
        Depth resolution (m).

    Returns
    -------
    tuple
        Derivatives of large and small eDNA concentrations.
    """
    dC_L_dz = np.gradient(C_L, dz)
    dC_S_dz = np.gradient(C_S, dz)

    return dC_L_dz, dC_S_dz


def predator_model(
    y_current,
    copepod_source,
    predator_dist,
    stomach_idx,
    predator_stomach,
    wv_minus_ws,
    wv,
    predation_efficiency,
    digestion_time_steps,
    precomputed_release_factors,
    k_plus_delta,
    k_array,
    delta_val,
    S_L,
    S_S,
    Nz,
    dz,
    kz,
):
    """
    Model for eDNA concentration dynamics with predator-prey interactions.

    Parameters
    ----------
    y_current : array_like
        Current state vector containing large and small eDNA concentrations.
    copepod_source : array_like
        Source term for copepod eDNA input.
    predator_dist : array_like
        Distribution of predators.
    stomach_idx : int
        Current stomach index.
    predator_stomach : array_like
        Predator stomach contents.
    wv_minus_ws : array_like
        Vertical advection minus settling rate.
    wv : array_like
        Vertical advection profile.
    predation_efficiency : float
        Efficiency of predation.
    digestion_time_steps : int
        Number of time steps for digestion.
    precomputed_release_factors : array_like
        Precomputed release factors.
    k_plus_delta : array_like
        Decay rate plus breakdown rate.
    k_array : array_like
        Decay rate profile.
    delta_val : float
        Breakdown rate from large to small eDNA.
    S_L : float
        Shedding rate for large eDNA.
    S_S : float
        Shedding rate for small eDNA.
    Nz : int
        Number of depth points.
    dz : float
        Depth resolution (m).
    kz : array_like
        Diffusivity profile.

    Returns
    -------
    tuple
        Rate of change of eDNA concentrations and updated predator stomach contents.
    """
    C_L = y_current[:Nz]
    C_S = y_current[Nz:]

    dC_L_dz, dC_S_dz = compute_derivatives(C_L, C_S, dz)
    diffusion_L = np.gradient(kz * dC_L_dz, dz)
    diffusion_S = np.gradient(kz * dC_S_dz, dz)

    # Calculate predation effects
    consumed_L, consumed_S = calculate_predation_interaction(
        copepod_source, predator_dist, C_L, C_S, predation_efficiency
    )

    # Update predator stomach
    updated_stomach = update_predator_stomach(
        consumed_L, consumed_S, stomach_idx, predator_stomach
    )

    # Calculate predator eDNA release
    released_L, released_S, updated_stomach = calculate_predator_release(
        updated_stomach,
        stomach_idx,
        predator_dist,
        digestion_time_steps,
        precomputed_release_factors,
    )

    dC_L_dt = (
        -wv_minus_ws * dC_L_dz
        + diffusion_L
        - k_plus_delta * C_L
        + copepod_source * S_L
        + released_L
    )

    dC_S_dt = (
        -wv * dC_S_dz
        + diffusion_S
        - k_array * C_S
        + delta_val * C_L
        + copepod_source * S_S
        + released_S
    )

    return np.concatenate([dC_L_dt, dC_S_dt]), updated_stomach


def precompute_release_factors(dt, decay_rate=0.1, max_iterations=100):
    """
    Precompute release factors for predator digestion.

    Parameters
    ----------
    dt : float
        Time step (s).
    decay_rate : float, optional
        Decay rate for release.
    max_iterations : int, optional
        Maximum number of iterations.

    Returns
    -------
    array_like
        Precomputed release factors.
    """
    factors = np.zeros(max_iterations)
    for i in range(max_iterations):
        time_since_digestion = i + 1
        factors[i] = decay_rate * np.exp(
            -decay_rate * time_since_digestion * dt / 3600
        )
    return factors


def run_predator_simulation(
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
    method: Literal["euler", "rk4"] = "euler",
):
    """
    Run a simulation of eDNA concentration dynamics with predator-prey interactions.

    Parameters
    ----------
    z : array_like
        Depth values (m).
    dz : float
        Depth resolution (m).
    total_steps : int
        Total number of time steps.
    dt : float
        Time step (s).
    copepod_pattern : array_like
        Distribution pattern of copepods over time and depth.
    predator_pattern : array_like
        Distribution pattern of predators over time and depth.
    predation_efficiency : float
        Efficiency of predation.
    digestion_time_steps : int
        Number of time steps for digestion.
    release_decay_rate : float
        Decay rate for release.
    release_duration_steps : int
        Number of time steps for release duration.
    wv : array_like
        Vertical advection profile.
    ws : float
        Settling rate.
    k : array_like
        Decay rate profile.
    delta : float
        Breakdown rate from large to small eDNA.
    S_L : float
        Shedding rate for large eDNA.
    S_S : float
        Shedding rate for small eDNA.
    y0 : array_like
        Initial state vector.
    method : str, optional
        Integration method. Options: "euler", "rk4".

    Returns
    -------
    array_like
        Array of eDNA concentrations over time and depth.
    """
    Nz = len(z)
    wv_minus_ws = wv - ws
    k_plus_delta = k + delta
    k_array = k
    delta_val = delta
    precomputed_release_factors = precompute_release_factors(
        dt, release_decay_rate, release_duration_steps
    )
    daily_steps = int(86400 / dt)

    concentration = []
    y = y0.copy()
    predator_stomach = np.zeros((digestion_time_steps, Nz, 2))

    for t in tqdm(
        range(total_steps), desc="Optimized simulation with predation"
    ):
        time_idx = t % daily_steps

        # Get current distributions
        copepod_source = copepod_pattern[time_idx]
        predator_dist = predator_pattern[time_idx]

        # Update stomach index
        stomach_idx = t % digestion_time_steps

        k1, predator_stomach = predator_model(
            y,
            copepod_source,
            predator_dist,
            stomach_idx,
            predator_stomach,
            wv_minus_ws,
            wv,
            predation_efficiency,
            digestion_time_steps,
            precomputed_release_factors,
            k_plus_delta,
            k_array,
            delta_val,
            S_L,
            S_S,
            Nz,
            dz,
            0,  # Assuming kz = 0 as in the original code
        )
        if method == "rk4":
            k2, stomach_temp2 = predator_model(
                y + 0.5 * k1,
                copepod_source,
                predator_dist,
                stomach_idx,
                predator_stomach,
                wv_minus_ws,
                wv,
                predation_efficiency,
                digestion_time_steps,
                precomputed_release_factors,
                k_plus_delta,
                k_array,
                delta_val,
                S_L,
                S_S,
                Nz,
                dz,
                0,
            )
            k3, stomach_temp3 = predator_model(
                y + 0.5 * k2,
                copepod_source,
                predator_dist,
                stomach_idx,
                stomach_temp2,
                wv_minus_ws,
                wv,
                predation_efficiency,
                digestion_time_steps,
                precomputed_release_factors,
                k_plus_delta,
                k_array,
                delta_val,
                S_L,
                S_S,
                Nz,
                dz,
                0,
            )
            k4, predator_stomach = predator_model(
                y + k3,
                copepod_source,
                predator_dist,
                stomach_idx,
                stomach_temp3,
                wv_minus_ws,
                wv,
                predation_efficiency,
                digestion_time_steps,
                precomputed_release_factors,
                k_plus_delta,
                k_array,
                delta_val,
                S_L,
                S_S,
                Nz,
                dz,
                0,
            )

        y += k1 if method == "euler" else (k1 + 2 * k2 + 2 * k3 + k4) / 6

        concentration.append(y.copy())

    return np.array(concentration)
