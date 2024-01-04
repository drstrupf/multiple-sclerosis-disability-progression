"""Annotate first progression event in a series of EDSS assessments.

Source publication: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6029149/


First progression
-----------------
An EDSS score counts as first progression if it is sufficiently
higher than the reference that applies at that time. The minimum 
required increase may depend on the reference. The original definition
in [1] also requires a certain amount of time to pass between
assessments, but it is unclear whether this refers to the previous
assessment or the previous reference.

The following options need to be covered:

- Minimal increase condition as function of the reference

- Progression only if a sufficient timedelta has passed since
  a) the last reference assessment
  b) the previous assessment

- Optional confirmation requirement, see next section


Confirmation
------------
For confirmation, we have the following options:

Which scores are relevant for confirmation?
  a) All scores within confirmation interval must satisfy condition
  b) Only the first score at t' >= t + delta must satisfy condition
  
What condition must they satisfy?
  c) Confirmation scores >= candidate score
  d) Confirmation scores >= reference + minimal increase

Confirmation minimal timedelta:
  a) Confirmed at x weeks: scores relevant for confirmation are
     those within x weeks and the first at t' >= t + delta
  b) Sustained and truncated: relevant score is the one after 
     the candidate, irrespective of delta, then truncate
  c) Sustained: relevant scores are all following scores until
     end of follow-up
  d) No confirmation required


References
----------
[1] Kappos L, Butzkueven H, Wiendl H, Spelman T, Pellegrini F,
Chen Y, Dong Q, Koendgen H, Belachew S, Trojano M; Tysabri® 
Observational Program (TOP) Investigators. “Greater sensitivity 
to multiple sclerosis disability worsening and progression 
events using a roving versus a fixed reference value in a 
prospective cohort study”. In: Multiple Sclerosis Journal
24.7 (2017), pp. 963-973. doi: 10.1177/1352458517709619

"""

import numpy as np
import pandas as pd


def is_above_progress_threshold(
    current_edss,
    reference_edss,
    increase_threshold=5.5,
    larger_increment_from_0=True,
):
    """Determine if a score meets the minimum increase condition.

    Standard definition:
    - baseline EDSS 0.0: increase of at least 1.5
    - baseline EDSS > 0.0 and <= 5.5: increase of at least 1.0
    - baseline EDSS >= 6.0: increase of at least 0.5

    This is reflected by the default arguments.

    For a minimal increase of 0.5 irrespective of baseline, choose
    larger_increment_from_0=False and increase_threshold=0.
    For a minimal increase of 1.0 irrespective of baseline, choose
    larger_increment_from_0=False and increase_threshold=10.0.

    Args:
      - current_edss: an EDSS score
      - reference_edss: the reference to which current_edss is compared
      - increase_threshold: the last EDSS where an increase of 1.0 is required
      - larger_increment_from_0: whether a baseline of 0.0 requires an increase of 1.5

    Returns:
      - bool: True if above progress threshold

    """
    if larger_increment_from_0 and reference_edss == 0:
        minimal_increase = 1.5
    else:
        if reference_edss <= increase_threshold:
            minimal_increase = 1
        else:
            minimal_increase = 0.5
    if current_edss >= reference_edss + minimal_increase:
        return True
    else:
        return False


def annotate_first_progression(
    follow_up_dataframe,
    edss_score_column_name,
    time_column_name,
    opt_increase_threshold=5.5,
    opt_larger_minimal_increase_from_0=True,
    opt_minimal_distance_time=0,
    opt_minimal_distance_type="reference",  # "reference" or "previous"
    opt_minimal_distance_backtrack_monotonic_decrease=True,
    opt_require_confirmation=False,
    opt_confirmation_time=-1,  # -1 for sustained over follow-up
    opt_confirmation_type="minimum",  # "minimum" or "monotonic"
    opt_confirmation_included_values="all",  # "last" or "all"
    reference_score_column_name="reference_edss_score",
    first_progression_flag_column_name="is_first_progression",
):
    """Flag assessments that qualify as first progression.

    This function returns a copy of the input dataframe with
    one additional column that flags the first progression
    event (bool, i.e. 'True' if progression, else 'False').

    The input dataframe is a series of EDSS assessments and
    their corresponding references (i.e. the output of the
    function baselines.annotate_baseline). It must have at
    least one column for the EDSS score, one for the timestamp,
    one for the reference score, and one for the reference
    score's timestamp.

    The column with the reference score timestamp, created
    by baselines.annotate_baseline, is reference_score_column_name
    + "_" + time_column_name by default.

    The data have to be ordered in time, and have only one EDSS
    score per timestep, i.e. time must be strictly monotonically
    increasing (but does not have to start at 0).

    The timestamp has to be provided as int or float, e.g.
    as 'days after baseline' or 'weeks after baseline'. Do
    not provide timestamps as date/datetime.

    Args:
      - follow_up_dataframe: a dataframe with an ordered EDSS follow-up and a column with reference scores
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

    # Check validity of arguments
    if opt_minimal_distance_type not in ["reference", "previous"]:
        raise ValueError(
            "Invalid minimal distance type. Options are 'reference' or 'previous'."
        )
    if opt_confirmation_type not in ["minimum", "monotonic"]:
        raise ValueError(
            "Invalid confirmation criterion type. Options are 'minimum' or 'monotonic'."
        )
    if opt_confirmation_included_values not in ["all", "last"]:
        raise ValueError(
            "Invalid confirmation scores type. Options are 'all' or 'last'."
        )
    if opt_minimal_distance_time < 0:
        raise ValueError("Invalid minimal distance time, must be >= 0.")
    if opt_require_confirmation and (
        (opt_confirmation_time != -1) and (opt_confirmation_time <= 0)
    ):
        raise ValueError(
            "Invalid input for confirmation interval. If confirmation required, choose -1 for sustained, or a duration > 0."
        )
    if (
        opt_require_confirmation
        and (opt_confirmation_time == -1)
        and (opt_confirmation_included_values == "last")
    ):
        raise ValueError(
            "Invalid confirmation requirements. For sustained progession, only the option 'all' is valid for included values."
        )

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

    # Prepare df to return in the end
    annotated_df = follow_up_dataframe.copy()
    annotated_df[first_progression_flag_column_name] = False

    # Control switch to break the loop
    is_first_progression = False

    # Now go through each row and check if it qualifies for new progression
    for i, row in follow_up_dataframe.iloc[1:].iterrows():
        # Keep track of the score
        current_edss = row[edss_score_column_name]

        # Keep track of time for minimal distance condition.
        # NOTE: We need a helper function if we want to adjust
        # for monotonic decrease when using a roving reference.
        # This is a tricky one, we have to go backwards from the
        # reference until we find a score in reference to which
        # the current score wouldn't be a progress anymore. Until
        # then, we adjust the timestamp of the rebaseline backwards...
        # Idea: we go back step by step and check if we are still low
        # enough for the current EDSS to be a progression and stop as
        # soon this condition is not fulfilled anymore.
        def _get_timestamp_of_rebaseline_in_monotonic_decrease():
            previous_scores_reverse = (
                follow_up_dataframe.loc[: i - 1]
                .sort_values(time_column_name, ascending=False)
                .copy()
            )
            # We need the timestamp of the current reference
            reference_time = row[reference_score_column_name + "_" + time_column_name]

            # We don't care about what happens between reference and progression,
            # we just need to know whether the reference was preceded by a monotonic
            # decrease that might be relevant for the minimum distance.
            previous_scores_reverse_before_reference = previous_scores_reverse[
                previous_scores_reverse[time_column_name] < reference_time
            ].copy()

            # Our current reference time stamp is that of the actual reference.
            timestamp_monotonic_rebase = row[
                reference_score_column_name + "_" + time_column_name
            ]

            #  Now we loop backwards over the scores before the reference.
            for _, subrow in previous_scores_reverse_before_reference.iterrows():
                if is_above_progress_threshold(
                    current_edss=current_edss,
                    reference_edss=subrow[edss_score_column_name],
                    larger_increment_from_0=opt_larger_minimal_increase_from_0,
                    increase_threshold=opt_increase_threshold,
                ):
                    timestamp_monotonic_rebase = subrow[time_column_name]
                else:
                    break

            return timestamp_monotonic_rebase

        current_timestamp = row[time_column_name]
        previous_timestamp = follow_up_dataframe.loc[i - 1][time_column_name]
        delta_time_to_previous = current_timestamp - previous_timestamp
        delta_time_to_reference = (
            current_timestamp
            - row[reference_score_column_name + "_" + time_column_name]
        )
        delta_time_to_reference_with_monotonic_backtracking = (
            current_timestamp - _get_timestamp_of_rebaseline_in_monotonic_decrease()
        )

        # We also need a helper function for confirmation.
        def _is_confirmed(current_edss, reference_edss):
            confirmed = False
            # If we require 'sustained', there must be at least one assessment
            # following the potential progression, and we consider the complete
            # remaining follow up.
            if opt_confirmation_time == -1:
                all_edss_within_interval = np.array(
                    follow_up_dataframe.loc[i + 1 :][edss_score_column_name]
                )
            # If we require confirmation after x days, we need the first
            # EDSS that is >= opt_confirmation_time after the current, and
            # all values in between.
            else:
                # Idea: take all assessments >= x after, then obtain the
                # index of the first entry, then for confirmation take all
                # rows from current up to and including this index.
                assessments_after_end_of_confirmation_interval = follow_up_dataframe[
                    follow_up_dataframe[time_column_name]
                    >= current_timestamp + opt_confirmation_time
                ].copy()

                # If there are no values that qualify, don't confirm.
                if len(assessments_after_end_of_confirmation_interval) == 0:
                    all_edss_within_interval = []

                else:
                    first_index_after_confirmation_interval = (
                        assessments_after_end_of_confirmation_interval.iloc[0].name
                    )
                    all_edss_within_interval = np.array(
                        follow_up_dataframe.loc[
                            i + 1 : first_index_after_confirmation_interval
                        ][edss_score_column_name]
                    )

            if len(all_edss_within_interval) > 0:
                if opt_confirmation_included_values == "last":
                    all_edss_within_interval = np.array([all_edss_within_interval[-1]])

                if opt_confirmation_type == "minimum":
                    if is_above_progress_threshold(
                        current_edss=min(all_edss_within_interval),
                        reference_edss=reference_edss,
                        increase_threshold=opt_increase_threshold,
                        larger_increment_from_0=opt_larger_minimal_increase_from_0,
                    ):
                        confirmed = True
                elif opt_confirmation_type == "monotonic":
                    if min(all_edss_within_interval) >= current_edss:
                        confirmed = True

            return confirmed

        # Now check progression
        is_first_progression = (
            is_above_progress_threshold(
                current_edss=current_edss,
                reference_edss=row[reference_score_column_name],
                increase_threshold=opt_increase_threshold,
                larger_increment_from_0=opt_larger_minimal_increase_from_0,
            )
            and (
                (
                    (opt_minimal_distance_type == "previous")
                    and (delta_time_to_previous >= opt_minimal_distance_time)
                )
                or (
                    (opt_minimal_distance_type == "reference")
                    and (not opt_minimal_distance_backtrack_monotonic_decrease)
                    and (delta_time_to_reference >= opt_minimal_distance_time)
                )
                or (
                    (opt_minimal_distance_type == "reference")
                    and (opt_minimal_distance_backtrack_monotonic_decrease)
                    and (
                        delta_time_to_reference_with_monotonic_backtracking
                        >= opt_minimal_distance_time
                    )
                )
            )
            and (
                (not opt_require_confirmation)
                or (
                    _is_confirmed(
                        current_edss=current_edss,
                        reference_edss=row[reference_score_column_name],
                    )
                )
            )
        )

        # Annotate result
        if is_first_progression:
            annotated_df.at[
                i, first_progression_flag_column_name
            ] = is_first_progression

        # Stop looping once we have found a first progression event
        if is_first_progression:
            break

    return annotated_df


if __name__ == "__main__":
    # Usage example
    import baselines

    example_follow_up_df = pd.DataFrame(
        [
            {"follow_up_id": 0, "days_after_baseline": 0, "edss_score": 5.0},
            {"follow_up_id": 0, "days_after_baseline": 12, "edss_score": 4.5},
            {"follow_up_id": 0, "days_after_baseline": 24, "edss_score": 4.5},
            {"follow_up_id": 0, "days_after_baseline": 36, "edss_score": 5.0},
            {"follow_up_id": 0, "days_after_baseline": 48, "edss_score": 4.5},
            {"follow_up_id": 0, "days_after_baseline": 60, "edss_score": 4.0},
            {"follow_up_id": 0, "days_after_baseline": 72, "edss_score": 3.5},
            {"follow_up_id": 0, "days_after_baseline": 84, "edss_score": 5.5},
            {"follow_up_id": 0, "days_after_baseline": 96, "edss_score": 5.5},
            {"follow_up_id": 0, "days_after_baseline": 108, "edss_score": 4.5},
        ]
    )

    example_follow_up_df_with_baseline = baselines.annotate_baseline(
        follow_up_dataframe=example_follow_up_df,
        baseline_type="roving",
        edss_score_column_name="edss_score",
        time_column_name="days_after_baseline",
        reference_score_column_name="reference_edss_score",
    )

    example_follow_up_df_with_baseline_and_progression = annotate_first_progression(
        follow_up_dataframe=example_follow_up_df_with_baseline,
        edss_score_column_name="edss_score",
        time_column_name="days_after_baseline",
        opt_increase_threshold=5.5,
        opt_larger_minimal_increase_from_0=True,
        opt_minimal_distance_time=0,
        opt_minimal_distance_type="reference",  # "reference" or "previous"
        opt_minimal_distance_backtrack_monotonic_decrease=True,
        opt_require_confirmation=False,
        opt_confirmation_time=-1,  # -1 for sustained over follow-up
        opt_confirmation_type="minimum",  # "minimum" or "monotonic"
        opt_confirmation_included_values="all",  # "last" or "all"
        reference_score_column_name="reference_edss_score",
        first_progression_flag_column_name="is_first_progression",
    )

    print(example_follow_up_df_with_baseline_and_progression)
