import numpy as np
import pandas as pd
import streamlit as st


from definitions import baselines, progression
from evaluation import survival, cohort


def follow_ups_to_annotated_progression(
    follow_up_dataframe,
    options_dictionary,
    id_column_name="follow_up_id",
    edss_score_column_name="edss_score",
    time_column_name="days_after_baseline",
    first_progression_flag_column_name="is_first_progression",
):
    """Get annotated progression for a bunch of follow-ups.

    See the docs of the functions called within for arg description.

    """
    # Some args are only internal, so set them here instead of passing them
    reference_score_column_name = "reference_edss_score"

    # Options from frontend.dynamic_progression_option_input_element()
    baseline_type = options_dictionary["baseline_type"]
    opt_increase_threshold = options_dictionary["opt_increase_threshold"]
    opt_larger_minimal_increase_from_0 = options_dictionary[
        "opt_larger_minimal_increase_from_0"
    ]
    opt_minimal_distance_time = options_dictionary["opt_minimal_distance_time"]
    opt_minimal_distance_type = options_dictionary["opt_minimal_distance_type"]
    opt_minimal_distance_backtrack_monotonic_decrease = options_dictionary[
        "opt_minimal_distance_backtrack_monotonic_decrease"
    ]
    opt_require_confirmation = options_dictionary["opt_require_confirmation"]
    opt_confirmation_time = options_dictionary["opt_confirmation_time"]
    opt_confirmation_type = options_dictionary["opt_confirmation_type"]
    opt_confirmation_included_values = options_dictionary[
        "opt_confirmation_included_values"
    ]

    # Annotate the baseline for each follow up
    follow_ups_with_baseline = cohort.annotate_baseline_cohort_level(
        follow_up_dataframe=follow_up_dataframe,
        baseline_type=baseline_type,
        id_column_name=id_column_name,
        edss_score_column_name=edss_score_column_name,
        time_column_name=time_column_name,
        reference_score_column_name=reference_score_column_name,
    )
    # Annotate the first progression event for each follow up
    follow_ups_with_baseline_and_progression = cohort.annotate_first_progression_cohort_level(
        follow_up_dataframe=follow_ups_with_baseline,
        id_column_name=id_column_name,
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
    return follow_ups_with_baseline_and_progression


def follow_ups_to_lifelines_input(
    follow_up_dataframe,
    options_dictionary,
    id_column_name="follow_up_id",
    edss_score_column_name="edss_score",
    time_column_name="days_after_baseline",
    additional_columns_to_drop=[],
    global_censoring=np.inf,
    duration_name="duration",
    observed_name="observed",
):
    """Get input for lifelines Kaplan-Meier fitter from a bunch of follow-ups.

    Eventually, lifelines only requires the 'duration' and 'observed' columns,
    so the names of intermediate step columns don't matter and are thus not
    passed as arguments.

    See the docs of the functions called within for arg description.

    """
    # Some args are only internal, so set them here instead of passing them
    reference_score_column_name = "reference_edss_score"
    first_progression_flag_column_name = "is_first_progression"
    progression_event_found_column_name = "progression"
    time_to_progression_column_name = "time_to_first_progression"
    time_to_last_before_progression_column_name = "time_to_last_before_progression"
    reference_for_progression_column_name = "reference_for_progression"
    progression_score_column_name = "progression_score"
    length_of_follow_up_column_name = "length_of_follow_up"

    # Options from frontend.dynamic_progression_option_input_element()
    baseline_type = options_dictionary["baseline_type"]
    opt_increase_threshold = options_dictionary["opt_increase_threshold"]
    opt_larger_minimal_increase_from_0 = options_dictionary[
        "opt_larger_minimal_increase_from_0"
    ]
    opt_minimal_distance_time = options_dictionary["opt_minimal_distance_time"]
    opt_minimal_distance_type = options_dictionary["opt_minimal_distance_type"]
    opt_minimal_distance_backtrack_monotonic_decrease = options_dictionary[
        "opt_minimal_distance_backtrack_monotonic_decrease"
    ]
    opt_require_confirmation = options_dictionary["opt_require_confirmation"]
    opt_confirmation_time = options_dictionary["opt_confirmation_time"]
    opt_confirmation_type = options_dictionary["opt_confirmation_type"]
    opt_confirmation_included_values = options_dictionary[
        "opt_confirmation_included_values"
    ]

    # Annotate the baseline for each follow up
    follow_ups_with_baseline = cohort.annotate_baseline_cohort_level(
        follow_up_dataframe=follow_up_dataframe,
        baseline_type=baseline_type,
        id_column_name=id_column_name,
        edss_score_column_name=edss_score_column_name,
        time_column_name=time_column_name,
        reference_score_column_name=reference_score_column_name,
    )
    # Annotate the first progression event for each follow up
    follow_ups_with_baseline_and_progression = cohort.annotate_first_progression_cohort_level(
        follow_up_dataframe=follow_ups_with_baseline,
        id_column_name=id_column_name,
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
    # Get time to progression
    follow_ups_time_to_progression = cohort.get_time_to_first_progression_cohort_level(
        follow_up_dataframe=follow_ups_with_baseline_and_progression,
        id_column_name=id_column_name,
        edss_score_column_name=edss_score_column_name,
        time_column_name=time_column_name,
        reference_score_column_name=reference_score_column_name,
        first_progression_flag_column_name=first_progression_flag_column_name,
        additional_columns_to_drop=additional_columns_to_drop,
        progression_event_found_column_name=progression_event_found_column_name,
        time_to_progression_column_name=time_to_progression_column_name,
        time_to_last_before_progression_column_name=time_to_last_before_progression_column_name,
        reference_for_progression_column_name=reference_for_progression_column_name,
        progression_score_column_name=progression_score_column_name,
        length_of_follow_up_column_name=length_of_follow_up_column_name,
    )
    # Add lifelines input colunns
    follow_ups_time_to_progression_for_lifelines = (
        cohort.get_lifelines_input_data_cohort_level(
            times_to_first_progression_dataframe=follow_ups_time_to_progression,
            time_to_progression_column_name=time_to_progression_column_name,
            length_of_follow_up_column_name=length_of_follow_up_column_name,
            global_censoring=global_censoring,
            duration_name=duration_name,
            observed_name=observed_name,
        )
    )
    return follow_ups_time_to_progression_for_lifelines


if __name__ == "__main__":
    pass
