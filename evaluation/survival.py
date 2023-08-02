"""Helper functions for survival analysis with lifelines.

Lifelines: https://lifelines.readthedocs.io/en/latest/

API references:
    - Kaplan-Meier: https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html#module-lifelines.fitters.kaplan_meier_fitter
    - Weibull: https://lifelines.readthedocs.io/en/latest/fitters/univariate/WeibullFitter.html#module-lifelines.fitters.weibull_fitter
    - Statistics: https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html#module-lifelines.statistics

"""

import numpy as np
import pandas as pd
import lifelines


def get_median_survival_time_kaplan_meier(
    times_to_event_df, durations_column_name="duration", observed_column_name="observed"
):
    """Get the number of events, event rate, and median survival with 95%CI with Kaplan-Meier.

    Args:
        - times_to_event_df: a dataframe with time to event/event observed data for a cohort
        - durations_column_name: the name of the column with duration (time to event/censoring)
        - observed_column_name: the name of the column with event observed flag

    Returns:
        - df: a dataframe with 1 row and the columns
            - n_events
            - event_rate
            - median_time_to_first_progression
            - median_time_to_first_progression_lower_95CI
            - median_time_to_first_progression_upper_95CI

    """
    kaplan_meier_fitter = lifelines.KaplanMeierFitter()
    durations = times_to_event_df[durations_column_name]
    observed = times_to_event_df[observed_column_name]
    kaplan_meier_fitter.fit(
        durations=durations,
        event_observed=observed,
    )

    # Count events, absolute and relative
    n_observations = len(durations)
    n_events_observed = sum(observed)
    event_rate = n_events_observed / n_observations

    # Get median survival times with 95% CI
    median_time_to_event = kaplan_meier_fitter.median_survival_time_
    median_time_to_event_ci = lifelines.utils.median_survival_times(
        kaplan_meier_fitter.confidence_interval_
    )

    return pd.DataFrame(
        [
            {
                "n_events": n_events_observed,
                "event_rate": event_rate,
                "median_time_to_first_progression": median_time_to_event,
                "median_time_to_first_progression_lower_95CI": median_time_to_event_ci.loc[
                    0.5
                ][
                    "KM_estimate_lower_0.95"
                ],
                "median_time_to_first_progression_upper_95CI": median_time_to_event_ci.loc[
                    0.5
                ][
                    "KM_estimate_upper_0.95"
                ],
            }
        ]
    )


if __name__ == "__main__":
    pass
