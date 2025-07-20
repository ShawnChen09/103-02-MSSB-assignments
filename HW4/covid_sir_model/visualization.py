import matplotlib.pyplot as plt
from config import STEP_PER_DAY


def plot_results(history):
    days = [i / STEP_PER_DAY for i in range(len(history["s"]))]

    plt.figure(figsize=(10, 6))
    plt.plot(days, history["s"], label="Susceptible", color="blue")
    plt.plot(days, history["i"], label="Infected", color="red")
    plt.plot(days, history["r"], label="Recovered", color="green")
    plt.plot(days, history["d"], label="Dead", color="black")

    plt.xlabel("Time (days)")
    plt.ylabel("Population")
    plt.title("Epidemic Curve for COVID-19 Simulation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    max_days = max(days)
    tick_spacing = (
        10
        if max_days > 100
        else 5
        if max_days > 30
        else 2
        if max_days > 10
        else 1
    )
    plt.xticks(range(0, int(max_days) + tick_spacing, tick_spacing))

    plt.tight_layout()
    plt.show()


def combine_plot_results(history_results):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for i, (args, history) in enumerate(history_results.items()):
        days = [i / STEP_PER_DAY for i in range(len(history["s"]))]
        ax = axes[i]
        ax.plot(days, history["s"], label="Susceptible", color="blue")
        ax.plot(days, history["i"], label="Infected", color="red")
        ax.plot(days, history["r"], label="Recovered", color="green")
        ax.plot(days, history["d"], label="Dead", color="black")
        ax.set_title(args)
        ax.grid(True, alpha=0.3)
        # if i == 2:
        #     ax.set_legend()

        max_days = max(days)
        tick_spacing = (
            10
            if max_days > 100
            else 5
            if max_days > 30
            else 2
            if max_days > 10
            else 1
        )
        axes[i].set_xticks(
            range(0, int(max_days) + tick_spacing, tick_spacing)
        )

    fig.supxlabel("Time (days)")
    fig.supylabel("Population")
    plt.tight_layout()
    plt.show()
