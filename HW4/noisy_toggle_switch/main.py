from analysis import plot_results
from simulation import run_simulations


def main():
    alpha_values = [5, 50, 500, 5000]
    u0 = [5, 0]
    num_runs = 100
    tend = 20

    results = run_simulations(
        alpha_values, u0=u0, tend=tend, num_runs=num_runs
    )

    plot_results(results, u0, num_runs)


if __name__ == "__main__":
    main()
