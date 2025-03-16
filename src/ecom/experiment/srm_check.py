"""Sample-ratio-mismatch check: is the observed split consistent with the design?"""

from __future__ import annotations

import pandas as pd

from ecom.stats.core import TestResult, srm_test

SRM_ALPHA = 0.01  # conventional stricter threshold: an SRM invalidates the whole test


def check_srm(df: pd.DataFrame, treatment_share: float = 0.5) -> TestResult:
    counts = df["group"].value_counts()
    observed = [int(counts.get("control", 0)), int(counts.get("treatment", 0))]
    res = srm_test(observed, [1 - treatment_share, treatment_share])
    res.alpha = SRM_ALPHA
    res.details["passed"] = bool(res.p_value >= SRM_ALPHA)
    return res
