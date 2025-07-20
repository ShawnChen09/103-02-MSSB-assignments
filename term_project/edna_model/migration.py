import numpy as np


def dvm(
    z,
    dz,
    total_steps,
    shallow_depth,
    deep_depth,
    ups_time,
    upe_time,
    downs_time,
    downe_time,
    layer_thickness=20,
):
    """
    Generate diel vertical migration (DVM) pattern.

    Parameters
    ----------
    z : array_like
        Depth values (m).
    dz : float
        Depth resolution (m).
    total_steps : int
        Total number of time steps for a full day.
    shallow_depth : float
        Shallow depth (m) where organisms stay during the day/night.
    deep_depth : float
        Deep depth (m) where organisms stay during the night/day.
    ups_time : float
        Start time (hour) for upward migration.
    upe_time : float
        End time (hour) for upward migration.
    downs_time : float
        Start time (hour) for downward migration.
    downe_time : float
        End time (hour) for downward migration.
    layer_thickness : float, optional
        Thickness of the layer where organisms are distributed.

    Returns
    -------
    array_like
        2D array of organism distribution over time and depth.
    """
    intensity = np.zeros((len(z), total_steps))

    shallow_depth = np.clip(shallow_depth, min(z), max(z))
    deep_depth = np.clip(deep_depth, min(z), max(z))

    layer_thickness = max(1.0, min(layer_thickness, (max(z) - min(z))))

    for t_idx in range(total_steps):
        t = t_idx / (total_steps / 24)

        if downs_time < t < downe_time:
            center_depth = shallow_depth + (deep_depth - shallow_depth) * (
                (t - downs_time) / (downe_time - downs_time)
            )
        elif ups_time < t < upe_time:
            center_depth = deep_depth - (deep_depth - shallow_depth) * (
                (t - ups_time) / (upe_time - ups_time)
            )
        else:
            if upe_time < downs_time:
                if t <= ups_time or t >= downe_time:
                    center_depth = deep_depth
                else:
                    center_depth = shallow_depth
            else:
                if t <= downs_time or t >= upe_time:
                    center_depth = shallow_depth
                else:
                    center_depth = deep_depth

        center_depth = np.clip(center_depth, min(z), max(z))

        center_idx = int(center_depth / dz)
        half_layer = int(layer_thickness / 2 / dz)

        layer_start_idx = max(0, center_idx - half_layer)
        layer_end_idx = min(len(z), center_idx + half_layer)

        if layer_end_idx > layer_start_idx:
            intensity[layer_start_idx:layer_end_idx, t_idx] = 1.0 / (
                layer_end_idx - layer_start_idx
            )
        else:
            intensity[center_idx, t_idx] = 1.0

    return intensity.T


def combine_dvm(patterns, weights):
    """
    Combine multiple DVM patterns with weights.

    Parameters
    ----------
    patterns : list of array_like
        List of DVM patterns to combine.
    weights : list of float
        Weights for each pattern.

    Returns
    -------
    array_like
        Combined DVM pattern.
    """
    weights = np.array(weights)
    if np.sum(weights) == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / np.sum(weights)

    combined = np.zeros_like(patterns[0])

    for pattern, weight in zip(patterns, weights, strict=False):
        combined += pattern * weight

    return combined
