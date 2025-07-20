import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncate a colormap to a specific range.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Colormap to truncate.
    minval : float, optional
        Minimum value for truncation.
    maxval : float, optional
        Maximum value for truncation.
    n : int, optional
        Number of colors in the new colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Truncated colormap.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def plot_concentration(
    z, days, concentration, vmin=-1, vmax=2, depth_range=None, cmap=None
):
    """
    Plot eDNA concentration over depth and time.

    Parameters
    ----------
    z : array_like
        Depth values (m).
    days : float
        Number of days in the simulation.
    concentration : array_like
        eDNA concentration array (log10 transformed).
    vmin : float, optional
        Minimum value for color scale.
    vmax : float, optional
        Maximum value for color scale.
    depth_range : tuple, optional
        Tuple of (min_depth, max_depth) to plot.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use. If None, a truncated 'jet' colormap is used.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    if cmap is None:
        base_cmap = plt.get_cmap("jet")
        cmap = truncate_colormap(base_cmap, 0, 0.65)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        concentration,
        aspect="auto",
        extent=[0, days, z[-1], z[0]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if depth_range:
        ax.set_ylim(depth_range)

    ax.invert_yaxis()
    ax.set_xlabel("Time (day)")
    ax.set_ylabel("Depth (m)")

    cbar = plt.colorbar(im)
    cbar.set_label("Log10(eDNA)")

    return fig


def plot_migration_pattern(z, pattern, days=1, title=None):
    """
    Plot migration pattern over depth and time.

    Parameters
    ----------
    z : array_like
        Depth values (m).
    pattern : array_like
        Migration pattern array.
    days : float, optional
        Number of days to plot.
    title : str, optional
        Title for the plot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    hours_per_day = 24
    total_steps = pattern.shape[0]
    steps_per_day = total_steps // days

    im = ax.imshow(
        pattern[:steps_per_day].T,
        aspect="auto",
        extent=[0, hours_per_day, z[-1], z[0]],
        cmap="viridis",
    )

    ax.invert_yaxis()
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel("Depth (m)")

    if title:
        ax.set_title(title)

    cbar = plt.colorbar(im)
    cbar.set_label("Relative Abundance")

    return fig
