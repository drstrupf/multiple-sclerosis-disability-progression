"""Functions to annotate baselines and first progression
events to a dataframe with multiple follow-ups.

The definitions for the baselines and progression provided
in definitions/baselines.py and definitions/progression.py
are for one single follow-up only. Thus, we need to loop over
all follow-ups when evaluating data from a cohort.

"""

import numpy as np
import pandas as pd

from evaluation import survival
from definitions import baselines, progression


def annotate_baseline_cohort_level(
    follow_up_dataframe,
    baseline_type,
    id_column_name,
    edss_score_column_name,
    time_column_name,
    reference_score_column_name="reference_edss_score",
):
    """Annotate baseline for multiple follow-ups at once.

    Takes the same arguments as annotate_baseline in
    definitions/baselines.py, and an additional arg for
    the column name where the follow-up ID is provided.

    Minimal required columns are EDSS, time, and ID. See
    the documentation for annotate_baseline for more info.

    NOTE: We don't do the list comprehension within pd.concat
    for performance reasons (would lead to a lot of copying...).

    Args:
        - follow_up_dataframe: a pandas dataframe
        - baseline_type: 'fixed' or 'roving'; how to define the baseline
        - id_column_name: the name of the column where the follow-up ID is specified
        - edss_score_column_name: the name of the column with the scores
        - time_column_name: the name of the column with the timestamps
        - reference_score_column_name: the name of the new column with the
          reference; default is 'reference_edss_score'. The column with the
          reference timestamp will be named reference_score_column_name +
          '_' + time_column name.

    Returns:
        - df: a copy of the input dataframe with the additional columns
          reference_score_column_name and reference_score_column_name +
          '_' + time_column name.

    """
    follow_up_dataframe_with_baseline_list = [
        baselines.annotate_baseline(
            follow_up_dataframe=follow_up_dataframe[
                follow_up_dataframe[id_column_name] == follow_up_id
            ],
            baseline_type=baseline_type,
            edss_score_column_name=edss_score_column_name,
            time_column_name=time_column_name,
            reference_score_column_name=reference_score_column_name,
        )
        for follow_up_id in follow_up_dataframe[id_column_name].drop_duplicates()
    ]
    follow_up_dataframe_with_baseline = pd.concat(
        follow_up_dataframe_with_baseline_list
    )
    return follow_up_dataframe_with_baseline


def annotate_first_progression_cohort_level(
    follow_up_dataframe,
    id_column_name,
    edss_score_column_name,
    time_column_name,
    opt_increase_threshold=5.5,
    opt_larger_minimal_increase_from_0=True,
    opt_minimal_distance_time=0,
    opt_minimal_distance_type="reference",
    opt_minimal_distance_backtrack_monotonic_decrease=True,
    opt_require_confirmation=False,
    opt_confirmation_time=-1,
    opt_confirmation_type="minimum",
    opt_confirmation_included_values="all",
    reference_score_column_name="reference_edss_score",
    first_progression_flag_column_name="is_first_progression",
):
    """Annotate first progression for multiple follow-ups at once.

    Takes the same arguments as annotate_first_progression in
    definitions/progression.py, and an additional arg for
    the column name where the follow-up ID is provided.

    Minimal required columns are EDSS, time, reference EDSS,
    reference EDSS timestamo, and ID. See the documentation
    for annotate_first_progression for more info.

    NOTE: We don't do the list comprehension within pd.concat
    for performance reasons (would lead to a lot of copying...).

    Args:
      - follow_up_dataframe: a dataframe with an ordered EDSS follow-up and a column with reference scores
      - id_column_name: the name of the column where the follow-up ID is specified
      - edss_score_column_name: column with EDSS score
      - time_column_name: column with timestamps
      - opt_increase_threshold: maximum EDSS for which an increase of 1.0 is required
      - opt_larger_minimal_increase_from_0: set to True if increment from 0.0 must be 1.5
      - opt_minimal_distance_time: minimum distance from reference or previous, 0 if none required
      - opt_minimal_distance_type: minimum distance to 'reference' or 'previous' assessment
      - opt_minimal_distance_backtrack_monotonic_decrease: in case of a progression after a monotonic
        decrease in roving baseline, measure timedelta from the first score which is low enough
      - opt_require_confirmation: set to True if confirmation required
      - opt_confirmation_time: length of confirmation interval, -1 for sustained over follow-up
      - opt_confirmation_type: whether confirmation values have to be >= minimal increase ('minimum') or >= current EDSS ('monotonic')
      - opt_confirmation_included_values: confirmation condition applies to 'all' values within interval or only 'last'.
      - reference_score_column_name: name of the column with the reference EDSS
      - first_progression_flag_column_name: name of the column that will indicate the first progression

    Returns:
      - df: a dataframe with additional boolean column first_progression_flag_column_name

    """
    # Annotate the first progression event for each follow up
    follow_up_dataframe_with_progression_list = [
        progression.annotate_first_progression(
            follow_up_dataframe=follow_up_dataframe[
                follow_up_dataframe[id_column_name] == follow_up_id
            ],
            edss_score_column_name=edss_score_column_name,
            time_column_name=time_column_name,
            opt_increase_threshold=opt_increase_threshold,
            opt_larger_minimal_increase_from_0=opt_larger_minimal_increase_from_0,
            opt_minimal_distance_time=opt_minimal_distance_time,
            opt_minimal_distance_type=opt_minimal_distance_type,
            opt_minimal_distance_backtrack_monotonic_decrease=opt_minimal_distance_backtrack_monotonic_decrease,
            opt_require_confirmation=opt_require_confirmation,
            opt_confirmation_time=opt_confirmation_time,
            opt_confirmation_type=opt_confirmation_type,
            opt_confirmation_included_values=opt_confirmation_included_values,
            reference_score_column_name=reference_score_column_name,
            first_progression_flag_column_name=first_progression_flag_column_name,
        )
        for follow_up_id in follow_up_dataframe[id_column_name].drop_duplicates()
    ]
    follow_up_dataframe_with_progression = pd.concat(
        follow_up_dataframe_with_progression_list
    )
    return follow_up_dataframe_with_progression


def get_time_to_first_progression_cohort_level(
    follow_up_dataframe,
    id_column_name,
    edss_score_column_name,
    time_column_name,
    reference_score_column_name="reference_edss_score",
    first_progression_flag_column_name="is_first_progression",
    additional_columns_to_drop=[],
    progression_event_found_column_name="progression",
    time_to_progression_column_name="time_to_first_progression",
    reference_for_progression_column_name="reference_for_progression",
    progression_score_column_name="progression_score",
    length_of_follow_up_column_name="length_of_follow_up",
):
    """Get time to progression for multiple follow-ups with annotated
    baseline and first progression.

    Calls survival.get_time_to_first_progression.

    Args:
        - follow_up_dataframe: a pandas dataframe with one follow-up, reference, progression
        - id_column_name: the name of the column where the follow-up ID is specified
        - edss_score_column_name: column with EDSS score
        - time_column_name: column with timestamps
        - reference_score_column_name: column with the reference score
        - first_progression_flag_column_name: column with the flag for first progression
        - additional_columns_to_drop: a list of columns to drop in the result
        - progression_event_found_column_name: name of the new colum to flag follow-ups with progression
        - time_to_progression_column_name: name of the new column with time to progression
        - reference_for_progression_column_name: name of the new column with the progression reference score
        - progression_score_column_name: name of the new column with the progression score
        - length_of_follow_up_column_name: name of the new column with the length of follow-up

    Returns:
        - df: a dataframe with 1 row and the following columns, in addition to any
          constant columns (e.g. the follow-up ID):
            - progression_event_found_column_name: bool, whether a progression event was found
            - time_to_progression_column_name: time to first progression event
            - reference_for_progression_column_name: progression was detected w.r.t. this reference
            - progression_score_column_name: the score at the assessment that counts as progression
            - length_of_follow_up_column_name: length of the follow-up period.

    """
    follow_up_dataframe_time_to_progression_list = [
        survival.get_time_to_first_progression(
            follow_up_dataframe=follow_up_dataframe[
                follow_up_dataframe[id_column_name] == follow_up_id
            ],
            edss_score_column_name=edss_score_column_name,
            time_column_name=time_column_name,
            reference_score_column_name=reference_score_column_name,
            first_progression_flag_column_name=first_progression_flag_column_name,
            additional_columns_to_drop=additional_columns_to_drop,
            progression_event_found_column_name=progression_event_found_column_name,
            time_to_progression_column_name=time_to_progression_column_name,
            reference_for_progression_column_name=reference_for_progression_column_name,
            progression_score_column_name=progression_score_column_name,
            length_of_follow_up_column_name=length_of_follow_up_column_name,
        )
        for follow_up_id in follow_up_dataframe[id_column_name].drop_duplicates()
    ]
    follow_up_dataframe_time_to_progression = pd.concat(
        follow_up_dataframe_time_to_progression_list
    )
    return follow_up_dataframe_time_to_progression


def get_lifelines_input_data_cohort_level(
    times_to_first_progression_dataframe,
    progression_event_found_column_name="progression",
    time_to_progression_column_name="time_to_first_progression",
    length_of_follow_up_column_name="length_of_follow_up",
    global_censoring=None,
    duration_name="duration",
    observed_name="observed",
):
    """Add lifelines-compatible 'duration' and 'observed' columns.
    
    Adds two columns to a dataframe where progression yes/no, time
    to progression, and length of follow-up are specified.

    Args:
        - times_to_first_progression_dataframe: a dataframe with times to progression
        - progression_event_found_column_name: the name of the column with the flag 'progression found'
        - time_to_progression_column_name: the column with time to progression
        - length_of_follow_up_column_name: the column with the length of follow-up
        - global_censoring: impose a global cutoff, default is None
        - duration_name: name of the dict entry for duration
        - observed_name: name of the dict entry for observed

    Returns:
        - dict: a dictionary with 'duration' and 'observed'

    """
    return_df = times_to_first_progression_dataframe.copy()
    return_df[[duration_name, observed_name]] = return_df.apply(
        lambda row: survival.get_lifelines_input_data(
            first_progression_flag=row[progression_event_found_column_name],
            time_to_first_progression=row[time_to_progression_column_name],
            length_of_follow_up=row[length_of_follow_up_column_name],
            global_censoring=global_censoring,
            duration_name=duration_name,
            observed_name=observed_name,
        ),
        axis=1,
        result_type="expand",
    )
    return return_df


if __name__ == "__main__":
    pass
