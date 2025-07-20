import numpy as np
from scipy.optimize import differential_evolution, minimize


def fit_migration_model(observed, z, dz, total_steps, num_patterns=2):
    """
    Decompose an observed migration pattern into a combination of multiple DVM patterns.

    Args:
        observed: Observed migration pattern data
        z: Depth array
        dz: Depth resolution
        total_steps: Total time steps
        num_patterns: Number of patterns to decompose into

    Returns:
        Dictionary containing decomposition results
    """
    global _observed_data, _z, _dz, _total_steps, _num_patterns
    _observed_data = observed
    _z = z
    _dz = dz
    _total_steps = total_steps
    _num_patterns = num_patterns

    def objective_function(params):
        penalty = 0
        patterns = []
        weights = []

        for i in range(num_patterns):
            idx = i * 6
            (
                shallow_depth,
                deep_depth,
                ups_time,
                downs_time,
                layer_thickness,
                weight,
            ) = params[idx : idx + 6]
            if shallow_depth >= deep_depth:
                penalty += 100 * (shallow_depth - deep_depth + 1) ** 2

            from edna_model.migration import dvm

            pattern = dvm(
                z,
                dz,
                total_steps,
                shallow_depth,
                deep_depth,
                ups_time,
                ups_time + 3,
                downs_time,
                downs_time + 3,
                layer_thickness,
            )

            patterns.append(pattern)
            weights.append(weight)

        from edna_model.migration import combine_dvm

        combined_model = combine_dvm(patterns, weights)

        return np.mean((combined_model - observed) ** 2) + penalty

    # Specify bounds
    bounds = []
    for _ in range(num_patterns):
        bounds.extend(
            [
                (min(z), max(z)),  # shallow_depth
                (min(z), max(z)),  # deep_depth
                (0, 21),  # ups_time
                (0, 21),  # downs_time
                (1, 200),  # layer_size
                (0.1, 1.0),  # weight
            ]
        )

    # Differential Evolution for global search
    de_result = differential_evolution(
        objective_function, bounds, maxiter=1000, popsize=15, seed=42, tol=1e-8
    )

    # Gradient descent for local refinement (Start from DE result)
    result = minimize(
        objective_function,
        de_result.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxiter": 10000,
            "maxfun": 50000,
            "disp": True,
        },
    )

    best_params = result.x
    patterns = []
    weights = []

    from edna_model.migration import combine_dvm, dvm

    for i in range(num_patterns):
        idx = i * 6
        param_set = best_params[idx : idx + 6]

        pattern = dvm(
            z,
            dz,
            total_steps,
            param_set[0],
            param_set[1],
            param_set[2],
            param_set[2] + 3,
            param_set[3],
            param_set[3] + 3,
            param_set[4],
        )
        patterns.append(pattern)
        weights.append(param_set[5])
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    combined_model = combine_dvm(patterns, weights)

    return {
        "parameters": best_params,
        "weights": weights,
        "individual_patterns": patterns,
        "combined_model": combined_model,
        "fit_error": result.fun,
        "observed": observed,
    }


def eval_model_fit(observed, predicted):
    """
    Evaluate the quality of model fit using various metrics.

    Args:
        observed: Observed data
        predicted: Predicted data from model

    Returns:
        Dictionary of evaluation metrics
    """
    mse = np.mean((observed - predicted) ** 2)
    rmse = np.sqrt(mse)

    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    ss_res = np.sum((observed - predicted) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    try:
        corr = np.corrcoef(observed.flatten(), predicted.flatten())[0, 1]
    except Exception:
        corr = 0

    return {
        "MSE": mse,
        "RMSE": rmse,
        "R-squared": r_squared,
        "Correlation": corr,
    }
