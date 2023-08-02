"""Helper functions for survival analysis.

Under construction... Untangle parts for publication
and parts for general evaluation...

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


def get_time_to_first_progression(
    follow_up_dataframe,
    edss_score_column_name,
    time_column_name,
    reference_score_column_name="reference_edss",
    first_progression_flag_column_name="is_first_progression",
    additional_columns_to_drop=["is_block_baseline"],
):
    """Get time to progression for a follow-up with annotated
    baseline and first progression.

    Args:
        - follow_up_dataframe: a pandas dataframe with one follow-up, reference, progression
        - edss_score_column_name: column with EDSS score
        - time_column_name: column with timestamps
        - reference_score_column_name: column with the reference score
        - first_progression_flag_column_name: column with the flag for first progression

    Returns:
        - df: a dataframe with 1 row and columns:
            - progression: bool, whether a progression event was found
            - time_to_first_progression: time to first progression event
            - reference_for_progression: progression was detected w.r.t. this reference
            - progression_score: the score at the assessment that counts as progression
            - length_of_follow_up: length of the follow-up period.

    """
    first_assessment_timestamp = follow_up_dataframe.iloc[0][time_column_name]
    last_assessment_timestamp = follow_up_dataframe.iloc[-1][time_column_name]
    length_of_follow_up = last_assessment_timestamp - first_assessment_timestamp

    progress_df = follow_up_dataframe[
        follow_up_dataframe[first_progression_flag_column_name]
    ]
    if len(progress_df) == 0:
        progression_event_found = False
        time_to_progression = np.NaN
        reference_for_progression = np.NaN
        progression_score = np.NaN
    else:
        progression_event_found = True
        time_to_progression = (
            progress_df.iloc[0][time_column_name] - first_assessment_timestamp
        )
        reference_for_progression = progress_df.iloc[0][reference_score_column_name]
        progression_score = progress_df.iloc[0][edss_score_column_name]

    left_df = (
        follow_up_dataframe.drop(
            columns=additional_columns_to_drop
            + [
                edss_score_column_name,
                time_column_name,
                reference_score_column_name,
                first_progression_flag_column_name,
                reference_score_column_name + "_" + time_column_name,
            ]
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    assert (
        len(left_df) == 1
    ), "Ambiguous column values, check if you drop all relevant columns!"

    left_df["progression"] = progression_event_found
    left_df["time_to_first_progression"] = time_to_progression
    left_df["reference_for_progression"] = reference_for_progression
    left_df["progression_score"] = progression_score
    left_df["length_of_follow_up"] = length_of_follow_up

    return left_df


def get_lifelines_input_data(
    first_progression_flag,
    time_to_first_progression,
    length_of_follow_up,
    global_censoring=None,
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
        - first_progression_flag: bool, whether a progression was observed
        - time_to_first_progression: time to first progression
        - length_of_follow_up: duration of follow-up
        - global_censoring: impose a global cutoff, default is None
        - duration_name: name of the dict entry for duration
        - observed_name: name of the dict entry for observed

    Returns:
        - dict: a dictionary with 'duration' and 'observed'

    """
    if global_censoring is None:
        observed = first_progression_flag
        if time_to_first_progression > 0:
            duration = time_to_first_progression
        else:
            duration = length_of_follow_up
    else:
        if (time_to_first_progression > 0) and (
            time_to_first_progression <= global_censoring
        ):
            duration = time_to_first_progression
            observed = True
        else:
            duration = global_censoring
            observed = False

    return {duration_name: duration, observed_name: observed}


if __name__ == "__main__":
    pass
