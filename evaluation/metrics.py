from math import asin, sqrt
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint


def wilson_ci(num_correct: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    low, high = proportion_confint(num_correct, n, alpha=alpha, method="wilson")
    return float(low), float(high)


def accuracy_with_ci(correct_series: pd.Series) -> Dict[str, float]:
    n = int(correct_series.shape[0])
    num_correct = int(correct_series.sum())
    acc = float(num_correct / n) if n > 0 else 0.0
    low, high = wilson_ci(num_correct, n)
    return {
        "n": n,
        "num_correct": num_correct,
        "accuracy": acc,
        "ci_low": low,
        "ci_high": high,
    }


def cohens_h(p1: float, p2: float) -> float:
    return 2.0 * (asin(sqrt(p1)) - asin(sqrt(p2)))


def mcnemar_test(y_true_a, y_true_b) -> Dict[str, float]:
    a = np.asarray(y_true_a)
    b = np.asarray(y_true_b)

    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))

    if (n01 + n10) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n01": n01, "n10": n10}

    statistic = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
    p_value = float(1 - chi2.cdf(statistic, df=1))
    return {"statistic": float(statistic), "p_value": p_value, "n01": n01, "n10": n10}


def compute_flops_per_token(active_params_b: float) -> float:
    return 2.0 * float(active_params_b) * 1e9
