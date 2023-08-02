import numpy as np
import pandas as pd
import streamlit as st


from definitions import baselines, progression
from evaluation import utils, survival


def get_lifelines_input_for_cohort(
    follow_up_dataframe,
    baseline_type,
    opt_increase_threshold,
    opt_larger_minimal_increase_from_0,
    opt_minimal_distance_time,
    opt_minimal_distance_type,
    opt_minimal_distance_backtrack_monotonic_decrease,
    opt_require_confirmation,
    opt_confirmation_time,
    opt_confirmation_type,
    opt_confirmation_included_values,
    id_column_name="follow_up_id",
    edss_score_column_name="edss_score",
    time_column_name="days_after_baseline",
):
    # Annotate the baseline for each follow up
    follow_up_dataframe_with_baseline_list = [
        baselines.annotate_baseline(
            follow_up_dataframe=follow_up_dataframe[
                follow_up_dataframe[id_column_name] == follow_up_id
            ],
            baseline_type=baseline_type,
            edss_score_column_name=edss_score_column_name,
            time_column_name=time_column_name,
            reference_score_column_name="reference_edss_score",
        )
        for follow_up_id in follow_up_dataframe[id_column_name].drop_duplicates()
    ]
    follow_up_dataframe_with_baseline = pd.concat(
        follow_up_dataframe_with_baseline_list
    )
    # Annotate the first progression event for each follow up
    follow_up_dataframe_with_baseline_and_progression_list = [
        progression.annotate_first_progression(
            follow_up_dataframe=follow_up_dataframe_with_baseline[
                follow_up_dataframe_with_baseline[id_column_name] == follow_up_id
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
            reference_score_column_name="reference_edss_score",
            first_progression_flag_column_name="is_first_progression",
        )
        for follow_up_id in follow_up_dataframe[id_column_name].drop_duplicates()
    ]
    follow_up_dataframe_with_baseline_and_progression = pd.concat(
        follow_up_dataframe_with_baseline_and_progression_list
    )
    # Get time to progression
    follow_up_dataframe_time_to_progression_list = [
        utils.get_time_to_first_progression(
            follow_up_dataframe=follow_up_dataframe_with_baseline_and_progression[
                follow_up_dataframe_with_baseline_and_progression["follow_up_id"]
                == follow_up_id
            ],
            edss_score_column_name="edss_score",
            time_column_name="days_after_baseline",
            reference_score_column_name="reference_edss_score",
            first_progression_flag_column_name="is_first_progression",
            additional_columns_to_drop=[],
        )
        for follow_up_id in follow_up_dataframe_with_baseline_and_progression[
            "follow_up_id"
        ].drop_duplicates()
    ]
    follow_up_dataframe_time_to_progression = pd.concat(
        follow_up_dataframe_time_to_progression_list
    )
    # Add lifelines input stuff...
    follow_up_dataframe_time_to_progression[
        ["duration", "observed"]
    ] = follow_up_dataframe_time_to_progression.apply(
        lambda row: utils.get_lifelines_input_data(
            first_progression_flag=row["progression"],
            time_to_first_progression=row["time_to_first_progression"],
            length_of_follow_up=row["length_of_follow_up"],
            global_censoring=None,
            duration_name="duration",
            observed_name="observed",
        ),
        axis=1,
        result_type="expand",
    )
    return follow_up_dataframe_time_to_progression


if __name__ == "__main__":
    pass
