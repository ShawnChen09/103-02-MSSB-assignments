import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_simulation_results(t, s0, s1):
    """
    Visualize the simulation results.

    Parameters:
        t : Time points
        s0 : Simulation results for first initial condition
        s1 : Simulation results for second initial condition
    """
    color_key = plt.get_cmap("rainbow_r")(np.linspace(0, 1, 3))
    color_hex = [
        matplotlib.colors.to_hex(color_key[i]) for i in range(len(color_key))
    ]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for i, si in enumerate(s0.T):
        ax[0].plot(t, si, color=color_hex[i])
    ss = np.round(s0[-1], 3)
    ax[0].set_title(f"s0={(0.3, 0.2, 0.1)}, ss={ss}")
    ax[0].grid()

    for i, (si, si_l) in enumerate(
        zip(s1.T, ["s1", "s2", "s3"], strict=False)
    ):
        ax[1].plot(t, si, color=color_hex[i], label=si_l)
    ss = np.round(s1[-1], 3)
    ax[1].set_title(f"s0={(6, 4, 4)}, ss={ss}")
    ax[1].grid()
    ax[1].legend(bbox_to_anchor=(1.25, 1))

    fig.supxlabel("Time (t)")
    fig.supylabel("s(t)")
    plt.show()
