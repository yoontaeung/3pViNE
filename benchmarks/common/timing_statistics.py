"""Sample dispersion and a normal-approximation 95% interval for the mean."""

import math
import statistics


def timing_statistics(values: list[float]) -> dict[str, float | int | None]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("finite timing samples are required")
    mean = statistics.fmean(values)
    variance = statistics.variance(values) if len(values) > 1 else None
    stddev = math.sqrt(variance) if variance is not None else None
    sem = stddev / math.sqrt(len(values)) if stddev is not None else None
    margin = statistics.NormalDist().inv_cdf(0.975) * sem if sem is not None else None
    return {
        "n": len(values), "min": min(values), "mean": mean, "max": max(values),
        "median": statistics.median(values), "variance": variance,
        "stddev": stddev, "sem": sem,
        "ci95_low": mean - margin if margin is not None else None,
        "ci95_high": mean + margin if margin is not None else None,
    }
