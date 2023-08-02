"""Helpers for publication/plots.

"""

import numpy as np
import pandas as pd


def create_definition_label(
    baseline_type,
    opt_increase_threshold,
    opt_larger_minimal_increase_from_0,
    opt_require_confirmation,
    opt_confirmation_time,
    opt_confirmation_included_values,
    opt_minimal_distance_time,
):
    """Build a descriptive string for a definition combination.

    Args:
        - baseline_type: type of reference used
        - opt_increase_threshold: threshold above which minimal increase is 0.5
        - opt_larger_minimal_increase_from_0: bool, whether increase form 0 is 1.5
        - opt_require_confirmation: bool, confirmation required or not
        - opt_confirmation_time: -1 if sustained, else any float > 0, interpreted as days
        - opt_confirmation_included_values: entire interval or last value only
        - opt_minimal_distance_time: minimal distance, any float >= 0, interpreted as days

    Returns:
        - str: a description of the definition

    """
    if not opt_larger_minimal_increase_from_0:
        if opt_increase_threshold == 0:
            increment_part = "Increase of 0.5"
        elif opt_increase_threshold == 10:
            increment_part = "Increase of 1.0"
        else:
            increment_part = (
                "Increase of 1.0 for reference < "
                + str(opt_increase_threshold + 0.5)
                + ", 0.5 for reference ≥ "
                + str(opt_increase_threshold + 0.5)
            )
    else:
        if opt_increase_threshold == 0:
            increment_part = "Increase of 1.5 from 0, otherwise 0.5"
        elif opt_increase_threshold == 10:
            increment_part = "Increase of 1.5 from 0, otherwise 1.0"
        else:
            increment_part = (
                "Increase of 1.5 from 0, 1.0 for reference < "
                + str(opt_increase_threshold + 0.5)
                + ", 0.5 for reference ≥ "
                + str(opt_increase_threshold + 0.5)
            )

    reference_part = " over " + baseline_type + " reference"

    if not opt_require_confirmation:
        confirmation_part = "unconfirmed"
    else:
        if opt_confirmation_included_values == "last":
            values_part = ", first assessment at t ≥ confirmation period only"
        elif opt_confirmation_included_values == "all":
            values_part = ", all values within confirmation period"

        if opt_confirmation_time == -1:
            confirmation_part = "sustained over entire follow-up" + values_part
        else:
            confirmation_part = (
                str(int(opt_confirmation_time / 7)) + " weeks confirmed" + values_part
            )

    if opt_minimal_distance_time == 0:
        minimal_distance_part = ""
    else:
        minimal_distance_part = (
            ", minimal distance to reference "
            + str(int(opt_minimal_distance_time / 7))
            + " weeks"
        )

    return (
        increment_part
        + reference_part
        + ", "
        + confirmation_part
        + minimal_distance_part
    )

if __name__ == "__main__":
    pass
