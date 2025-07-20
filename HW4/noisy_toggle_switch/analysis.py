import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter


def sample_and_average(results, alpha, num_points=100):
    """
    Sample and average results for a given alpha value.

    Parameters:
    -----------
    results : dict
        Simulation results
    alpha : float
        Alpha value to analyze
    num_points : int, optional
        Number of sample points

    Returns:
    --------
    tuple
        (sample_times, mean_p1, mean_p2, std_p1, std_p2)
    """
    max_time = max(ts[-1] for ts, _ in results[alpha])
    sample_times = np.linspace(0, max_time, num_points)

    all_p1 = np.zeros((len(results[alpha]), len(sample_times)))
    all_p2 = np.zeros((len(results[alpha]), len(sample_times)))

    for i, (ts, us) in enumerate(results[alpha]):
        for j, t in enumerate(sample_times):
            if t <= ts[-1]:
                idx = np.searchsorted(ts, t)
                if idx == len(ts):
                    idx = len(ts) - 1
                all_p1[i, j] = us[idx, 0]
                all_p2[i, j] = us[idx, 1]
            else:
                all_p1[i, j] = us[-1, 0]
                all_p2[i, j] = us[-1, 1]

    mean_p1 = np.mean(all_p1, axis=0)
    mean_p2 = np.mean(all_p2, axis=0)
    std_p1 = np.std(all_p1, axis=0)
    std_p2 = np.std(all_p2, axis=0)

    return sample_times, mean_p1, mean_p2, std_p1, std_p2


def plot_results(results, u0, num_runs):
    """
    Plot the results of the simulations.

    Parameters:
    -----------
    results : dict
        Simulation results
    u0 : list
        Initial state [N1, N2]
    num_runs : int
        Number of runs for each alpha value

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2)

    for i, alpha in enumerate(results.keys()):
        ax = fig.add_subplot(gs[i // 2, i % 2])

        for _, (ts, us) in enumerate(results[alpha]):
            ax.plot(ts, us[:, 0], "b-", alpha=0.2, linewidth=0.5)
            ax.plot(ts, us[:, 1], "r-", alpha=0.2, linewidth=0.5)

        sample_times, mean_p1, mean_p2, std_p1, std_p2 = sample_and_average(
            results, alpha
        )

        ax.plot(sample_times, mean_p1, "b-", linewidth=2, label="P1 average")
        ax.plot(sample_times, mean_p2, "r-", linewidth=2, label="P2 average")
        ax.fill_between(
            sample_times,
            mean_p1 - std_p1,
            mean_p1 + std_p1,
            color="b",
            alpha=0.2,
        )
        ax.fill_between(
            sample_times,
            mean_p2 - std_p2,
            mean_p2 + std_p2,
            color="r",
            alpha=0.2,
        )

        ax.set_title(f"α = {alpha}", fontsize=14)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("# of molecules", fontsize=12)
        ax.legend(fontsize=10, loc="upper right")

        if alpha >= 500:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())

        half_idx = len(sample_times) // 2
        late_mean_p1 = np.mean(mean_p1[half_idx:])
        late_mean_p2 = np.mean(mean_p2[half_idx:])
        late_std_p1 = np.mean(std_p1[half_idx:])
        late_std_p2 = np.mean(std_p2[half_idx:])

        stat_text = (
            f"Mean P1: {late_mean_p1:.1f} ± {late_std_p1:.1f}\n"
            + f"Mean P2: {late_mean_p2:.1f} ± {late_std_p2:.1f}"
        )

        ax.text(
            0.05,
            0.95,
            stat_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

    plt.tight_layout()
    plt.suptitle(
        f"Toggle Switch: Time Series for α Values (u0=[{u0[0]}, {u0[1]}], #runs={num_runs})",
        fontsize=16,
        y=1.02,
    )
    plt.show()

    return fig
