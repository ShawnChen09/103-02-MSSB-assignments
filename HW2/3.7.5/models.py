from typing import Annotated

import numpy as np


def mm_model(
    s: float,
    vmax: Annotated[list[float], 3] | Annotated[tuple[float], 3],
    km: Annotated[list[float], 3] | Annotated[tuple[float], 3],
) -> Annotated[tuple[float], 3]:
    """
    Defines the Michaelis-Menten equation for the rate of change of substrates.
    The equation is defined as vi = V_max * si / (Km + si) for i in (1, 2, 3).

    Parameters:
        s : The concentration of the substrate at a given time.
        vmax : The theoretical maximum velocity for each reaction.
        km : The Michaelis constant for each reaction (the concentration of substrate at half-maximal velocity).

    Returns:
        A tuple representing the velocities of the substrate at a given time (v1, v2, v3).
    """
    return np.array([(vmax[i] * s[i]) / (km[i] + s[i]) for i in range(3)])


def ma_model(s: float, k) -> float:
    """
    Defines the first-order Mass-action equation for the rate of change of substrates.
    The equation is defined as vi = ki * si for i in (1, 2, 3).

    Parameters:
        s : The concentration of the substrate at a given time.
        k : The rate constant for each reaction.

    Returns:
        A tuple representing the velocities of the substrate at a given time (v1, v2, v3).
    """
    return np.array([s[i] * k[i] for i in range(3)])
