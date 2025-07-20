import os
import sys

import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edna_model.migration import combine_dvm, dvm
from edna_model.pattern_decomposition import (
    eval_model_fit,
    fit_migration_model,
)
from edna_model.pattern_visualization import visualize_migration_patterns

if __name__ == "__main__":
    # Setup parameters
    day = 1
    z_max = 600.0
    dz = 10
    z = np.arange(0, z_max + dz, dz)

    T_total = (
        day * 24 * 3600.0
    )  # Total Time (s): 1 day * 24 hours/day * 3600 seconds/hour
    dt = 360.0  # Time step (s)
    times = int(T_total / dt)

    pattern1 = dvm(
        z,
        dz,
        times,
        shallow_depth=100,
        deep_depth=500,
        ups_time=18,
        upe_time=21,
        downs_time=6,
        downe_time=9,
        layer_thickness=100,
    )

    pattern2 = dvm(
        z,
        dz,
        times,
        shallow_depth=100,
        deep_depth=500,
        ups_time=6,
        upe_time=9,
        downs_time=18,
        downe_time=21,
        layer_thickness=100,
    )

    combined_data = combine_dvm([pattern1, pattern2], [0.7, 0.3])

    results = fit_migration_model(combined_data, z, dz, times, num_patterns=2)
    results["observed_data"] = combined_data

    fit_quality = eval_model_fit(combined_data, results["combined_model"])
    for metric, value in fit_quality.items():
        print(f"  {metric}: {value:.4f}")

    visualize_migration_patterns(z, results)
