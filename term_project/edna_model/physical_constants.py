import numpy as np


def kappa_profile(z, season="summer", L_scale=2.0):
    """
    Calculate diffusivity (kz_SEASON) profile.

    Parameters
    ----------
    z : array_like
        Depth values (m).
    season : str, optional
        Season to calculate diffusivity for. Options: "summer", "spring", "fall", "winter".
    L_scale : float, optional
        Length scale for the transition between surface and deep diffusivity.

    Returns
    -------
    array_like
        Diffusivity values at each depth.
    """
    ml_depths = {"summer": 30, "spring": 70, "fall": 90, "winter": 140}
    ml_depth = ml_depths.get(season, 50)

    kappa_surface = 1e-3
    kappa_deep = 1e-5

    kappa = kappa_deep + (kappa_surface - kappa_deep) * 0.5 * (
        1 - np.tanh((z - ml_depth) / L_scale)
    )
    return kappa


def vertical_advection_profile(z, w_max=1e-4, direction="down"):
    """
    Calculate vertical advection (wv) profile.

    Parameters
    ----------
    z : array_like
        Depth values (m).
    w_max : float, optional
        Maximum vertical advection velocity (m/s).
    direction : str, optional
        Direction of advection. Options: "up", "down".

    Returns
    -------
    array_like
        Vertical advection values at each depth.
    """
    if direction not in ["up", "down"]:
        raise ValueError("Direction must be 'up' or 'down'")
    if w_max > 0 and direction == "down":
        w_max *= -1

    w = np.zeros_like(z)

    for i, zz in enumerate(z):
        if 0 <= zz <= 200:
            # 0 ~ 200 m: Linear increase from 0 to w_max
            w[i] = (w_max / 200.0) * zz
        elif 200 < zz <= 400:
            # 200 ~ 400 m: Linear decrease from w_max to 0
            w[i] = w_max - ((zz - 200) * w_max / 200.0)
        else:
            # Over 400 m, set to 0
            w[i] = 0.0

    return w


def decay_rate_from_temp(T):
    """
    Calculate decay rate based on temperature.

    Parameters
    ----------
    T : float or array_like
        Temperature (°C).

    Returns
    -------
    float or array_like
        Decay rate.
    """
    return 0.05 + 0.0014 * T
