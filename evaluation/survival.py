"""Helper functions for survival analysis with lifelines.

Lifelines: https://lifelines.readthedocs.io/en/latest/

"""

import numpy as np
import pandas as pd
import lifelines


def get_time_to_first_progression(
    follow_up_dataframe,
    edss_score_column_name,
    time_column_name,
    reference_score_column_name="reference_edss_score",
    first_progression_flag_column_name="is_first_progression",
    additional_columns_to_drop=[],
    progression_event_found_column_name="progression",
    time_to_progression_column_name="time_to_first_progression",
    time_to_last_before_progression_column_name="time_to_last_before_progression",
    reference_for_progression_column_name="reference_for_progression",
    progression_score_column_name="progression_score",
    length_of_follow_up_column_name="length_of_follow_up",
):
    """Get time to progression for a follow-up with annotated
    baseline and first progression.

    Args:
        - follow_up_dataframe: a pandas dataframe with one follow-up, reference, progression
        - edss_score_column_name: column with EDSS score
        - time_column_name: column with timestamps
        - reference_score_column_name: column with the reference score
        - first_progression_flag_column_name: column with the flag for first progression
        - additional_columns_to_drop: a list of columns to drop in the result
        - progression_event_found_column_name: name of the new colum to flag follow-ups with progression
        - time_to_progression_column_name: name of the new column with time to progression
        - time_to_last_before_progression_column_name: name of the new column with time to last assessment before progression
        - reference_for_progression_column_name: name of the new column with the progression reference score
        - progression_score_column_name: name of the new column with the progression score
        - length_of_follow_up_column_name: name of the new column with the length of follow-up

    Returns:
        - df: a dataframe with 1 row and the following columns, in addition to any
          constant columns (e.g. the follow-up ID):
            - progression_event_found_column_name: bool, whether a progression event was found
            - time_to_progression_column_name: time to first progression event
            - time_to_last_before_progression_column_name: time to last assessment before progression
            - reference_for_progression_column_name: progression was detected w.r.t. this reference
            - progression_score_column_name: the score at the assessment that counts as progression
            - length_of_follow_up_column_name: length of the follow-up period.

    """
    # Check data type of timestamp column
    assert pd.api.types.is_numeric_dtype(
        follow_up_dataframe[time_column_name]
    ), "Timestamps must be numeric, e.g. an integer number of days after baseline."
    assert pd.api.types.is_numeric_dtype(
        follow_up_dataframe[reference_score_column_name + "_" + time_column_name]
    ), "Timestamps must be numeric, e.g. an integer number of days after baseline."
    # Assert that the input data are well ordered
    assert (
        follow_up_dataframe[time_column_name].is_monotonic_increasing
        and follow_up_dataframe[time_column_name].is_unique
    ), "Input data are not well ordered or contain ambiguous timestamps."

    first_assessment_timestamp = follow_up_dataframe.iloc[0][time_column_name]
    last_assessment_timestamp = follow_up_dataframe.iloc[-1][time_column_name]
    length_of_follow_up = last_assessment_timestamp - first_assessment_timestamp

    progress_df = follow_up_dataframe[
        follow_up_dataframe[first_progression_flag_column_name]
    ]
    if len(progress_df) == 0:
        progression_event_found = False
        time_to_progression = np.NaN
        time_to_last_before_progression = np.NaN
        reference_for_progression = np.NaN
        progression_score = np.NaN
    else:
        progression_event_found = True
        time_at_progression = progress_df.iloc[0][time_column_name]
        time_to_progression = time_at_progression - first_assessment_timestamp
        time_at_last_before_progression = follow_up_dataframe[
            follow_up_dataframe[time_column_name] < time_at_progression
        ].iloc[-1][time_column_name]
        time_to_last_before_progression = (
            time_at_last_before_progression - first_assessment_timestamp
        )
        reference_for_progression = progress_df.iloc[0][reference_score_column_name]
        progression_score = progress_df.iloc[0][edss_score_column_name]

    # Let's keep all ID columns etc., and just drop the
    # time-dependent stuff, and optionally some more...
    columns_to_drop = additional_columns_to_drop + [
        edss_score_column_name,
        time_column_name,
        reference_score_column_name,
        first_progression_flag_column_name,
        reference_score_column_name + "_" + time_column_name,
    ]
    columns_to_keep = [
        column
        for column in follow_up_dataframe.columns
        if column not in columns_to_drop
    ]
    return_df_base = (
        follow_up_dataframe[columns_to_keep]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )
    # Then we add the time to progression etc.
    return_df_left = pd.DataFrame(
        [
            {
                progression_event_found_column_name: progression_event_found,
                time_to_progression_column_name: time_to_progression,
                time_to_last_before_progression_column_name: time_to_last_before_progression,
                reference_for_progression_column_name: reference_for_progression,
                progression_score_column_name: progression_score,
                length_of_follow_up_column_name: length_of_follow_up,
            }
        ]
    )
    if return_df_base.empty:
        return_df = return_df_left
    else:
        return_df = pd.concat([return_df_base, return_df_left], axis=1)

    assert (
        len(return_df) == 1
    ), "Ambiguous column values, check if you drop all relevant columns!"

    return return_df


# Get input data for standard right-censored survival analysis
def get_lifelines_input_data(
    time_to_first_progression,
    length_of_follow_up,
    global_censoring=np.inf,
    duration_name="duration",
    observed_name="observed",
):
    """Format data for use with the lifelines library.

    We use lifelines (https://lifelines.readthedocs.io/en/latest/)
    for survival analyses. Most fitters (e.g. Kaplan-Meier) require
    the data points 'duration' (time to event or censoring) and
    'observed' (bool, whether an event was observed). We add these
    columns to the dataframe obtained using get_time_to_first_progression.

    Args:
        - time_to_first_progression: time to first progression
        - length_of_follow_up: duration of follow-up
        - global_censoring: impose a global cutoff, default is np.inf
        - duration_name: name of the dict entry for duration
        - observed_name: name of the dict entry for observed

    Returns:
        - dict: a dictionary with 'duration' and 'observed'

    """
    # Check if the input is valid
    if time_to_first_progression > 0:
        assert (
            time_to_first_progression <= length_of_follow_up
        ), "Invalid input, time to progression must not be larger than length of follow-up!"
    maximum_follow_up = min(length_of_follow_up, global_censoring)
    if (time_to_first_progression > 0) and (
        time_to_first_progression <= maximum_follow_up
    ):
        observed = True
        duration = time_to_first_progression
    else:
        observed = False
        duration = maximum_follow_up
    return {duration_name: duration, observed_name: observed}


# Survival for right-censored data, Kaplan-Meier
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


# Get input data for interval-censored survival analysis
def get_lifelines_input_data_interval_censored(
    time_to_first_progression,
    time_to_last_before_progression,
    length_of_follow_up,
    global_censoring=np.inf,
    lower_bound_name="lower_bound",
    upper_bound_name="upper_bound",
):
    """Format data for use with the lifelines library.

    We use lifelines (https://lifelines.readthedocs.io/en/latest/)
    for survival analyses. Interval censored fitting requires a lower
    and an upper bound for the time to event. We add these columns to
    the dataframe obtained using get_time_to_first_progression.

    Args:
        - time_to_first_progression: time to first progression
        - time_to_last_before_progression: time to last assessment before progression
        - length_of_follow_up: duration of follow-up
        - global_censoring: impose a global cutoff, default is None
        - lower_bound_name: name of the dict entry for the lower bound
        - upper_bound_name: name of the dict entry for the upper bound

    Returns:
        - dict: a dictionary with lower bound and upper bound

    """
    # Check if the input is valid
    if time_to_first_progression > 0:
        assert (
            time_to_first_progression <= length_of_follow_up
        ), "Invalid input, time to progression must not be larger than length of follow-up!"
    maximum_follow_up = min(length_of_follow_up, global_censoring)
    if (time_to_first_progression > 0) and (
        time_to_first_progression <= maximum_follow_up
    ):
        lower_bound = time_to_last_before_progression
        upper_bound = time_to_first_progression
    else:
        lower_bound = maximum_follow_up
        upper_bound = np.inf
    return {lower_bound_name: lower_bound, upper_bound_name: upper_bound}


# Survival for right-censored data, Weibull
def get_median_survival_time_weibull_interval(
    times_to_event_df,
    lower_bound_column_name="lower_bound",
    upper_bound_column_name="upper_bound",
):
    """Weibull fitter, interval censored, TBD."""
    weibull_fitter = lifelines.WeibullFitter()
    lower_bound = times_to_event_df[lower_bound_column_name]
    upper_bound = times_to_event_df[upper_bound_column_name]
    weibull_fitter.fit_interval_censoring(
        lower_bound=lower_bound, upper_bound=upper_bound
    )

    # Count events, absolute and relative
    n_observations = len(lower_bound)
    n_events_observed = len(upper_bound[upper_bound < np.inf])
    event_rate = n_events_observed / n_observations

    # Get median survival times with 95% CI
    median_time_to_event = weibull_fitter.median_survival_time_
    median_time_to_event_ci = lifelines.utils.median_survival_times(
        weibull_fitter.confidence_interval_survival_function_
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
                    "Weibull_estimate_lower_0.95"
                ],
                "median_time_to_first_progression_upper_95CI": median_time_to_event_ci.loc[
                    0.5
                ][
                    "Weibull_estimate_upper_0.95"
                ],
            }
        ]
    )


if __name__ == "__main__":
    pass
