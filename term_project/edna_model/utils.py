import numpy as np
import pandas as pd


def load_decay_rate(filepath):
    """
    Load decay rate data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing decay rate data.
    """
    return pd.read_csv(filepath)


def get_center_depth_trajectory(pattern, z):
    """
    Find the center depth (depth of maximum concentration) at each time step.

    Parameters
    ----------
    pattern : array_like
        2D array of organism distribution over time and depth.
    z : array_like
        Depth values (m).

    Returns
    -------
    array_like
        Center depth trajectory.
    """
    center_depths = []
    for t in range(pattern.shape[0]):
        max_idx = np.argmax(pattern[t, :])
        center_depths.append(z[max_idx])
    return np.array(center_depths)


def setup_simulation_parameters(days=4, z_max=1500.0, dz=0.5):
    """
    Set up basic simulation parameters.

    Parameters
    ----------
    days : int, optional
        Number of days to simulate.
    z_max : float, optional
        Maximum depth (m).
    dz : float, optional
        Depth resolution (m).

    Returns
    -------
    dict
        Dictionary containing simulation parameters.
    """
    z = np.arange(0, z_max + dz, dz)
    Nz = len(z)

    T_total = days * 24 * 3600.0  # Total Time (s)
    dt = 10.0  # Time step (s)
    total_steps = int(T_total / dt)

    # Initialize concentrations
    C_L = np.zeros(Nz)
    C_S = np.zeros(Nz)
    y0 = np.concatenate([C_L, C_S])

    return {
        "z": z,
        "Nz": Nz,
        "dz": dz,
        "T_total": T_total,
        "dt": dt,
        "total_steps": total_steps,
        "y0": y0,
    }
