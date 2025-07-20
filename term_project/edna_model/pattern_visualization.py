import matplotlib.pyplot as plt
import numpy as np


def visualize_migration_patterns(z, results):
    """
    Visualize the decomposed migration patterns.

    Args:
        z: Depth array
        results: Results from pattern decomposition
    """
    n_patterns = len(results["individual_patterns"])
    fig, axes = plt.subplots(
        n_patterns + 2,
        1,
        figsize=(12, 4 * (n_patterns + 2)),
        constrained_layout=True,
    )

    im = axes[0].imshow(
        results["observed"].T,
        aspect="auto",
        origin="lower",
        extent=[0, 24, min(z), max(z)],
        cmap="viridis",
    )
    axes[0].set_title("Observed data")
    axes[0].set_xlabel("Time (hrs)")
    axes[0].set_ylabel("Depth (m)")
    plt.colorbar(im, ax=axes[0])

    im = axes[1].imshow(
        results["combined_model"].T,
        aspect="auto",
        origin="lower",
        extent=[0, 24, min(z), max(z)],
        cmap="viridis",
    )
    axes[1].set_title("Combined Model")
    axes[1].set_xlabel("Time (hrs)")
    axes[1].set_ylabel("Depth (m)")
    plt.colorbar(im, ax=axes[1])

    for i, pattern in enumerate(results["individual_patterns"]):
        im = axes[i + 2].imshow(
            pattern.T,
            aspect="auto",
            origin="lower",
            extent=[0, 24, min(z), max(z)],
            cmap="viridis",
        )
        axes[i + 2].set_title(
            f"Mode {i + 1} (weights: {results['weights'][i]:.2f})"
        )
        axes[i + 2].set_xlabel("Time (hrs)")
        axes[i + 2].set_ylabel("Depth (m)")
        plt.colorbar(im, ax=axes[i + 2])

    plt.show()
