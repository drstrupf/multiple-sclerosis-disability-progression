"""TBD"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class EDSSProgression:
    """EDSS progression event detection and classification.

    This class provides the functionality to annotate EDSS
    disability accrual in a follow-up containing EDSS scores
    and a timestamp for each score.

    Definition options are specified when instantiating the
    class, with our recommendations as default options.

    Default options: 6 months (180 days) all-confirmed (minimum) 
    with respect to a 30-days all-confirmed roving reference, no
    minimal distance requirement, no event merging, no left-hand
    tolerance or right-hand constraints on confirmation time for
    reference or event confirmation, minimum increase + 1.5 for
    score = 0, + 1.0 for scores < 5.5, and + 1 for scores >= 5.5,
    RAW window 30 days pre- and 90 days post-relapse.

    The effects of individual parameter choices and parameter
    combinations are showcased in the methods.ipynb notebook
    in the repo's main folder.

    To annotate disability accrual events in a follow-up,
    create an instance of EDSSProgression, then use the
    add_progression_events_to_follow_up method that takes a
    pandas dataframe with a follow-up (at least two columns,
    one for the timestamps and one for the EDSS scores) and
    a list of relapse timestamps to get the annotated dataframe.

    See the tutorial.ipynb notebook in the repo's main folder
    for some input data format and usage examples.

    """

    # Options for undefined worsening events
    undefined_progression: str = "re-baselining only"  # or "never", "all", "end"
    undefined_progression_wrt_raw_pira_baseline: str = (
        "any"  # or "equal or greater", "greater only"
    )
    # Search mode options
    return_first_event_only: bool = False
    merge_continuous_events: bool = False
    continuous_events_max_repetition_time: int = 30
    continuous_events_max_merge_distance: int = (
        np.inf
    )  # be more conservative for sparse follow-ups!
    # Baseline options
    opt_baseline_type: str = "roving"
    opt_roving_reference_require_confirmation: bool = True
    opt_roving_reference_confirmation_time: float = 30 # 0.5 would amount to next confirmed
    opt_roving_reference_confirmation_included_values: str = "all"  # "last" or "all"
    opt_roving_reference_confirmation_time_right_side_max_tolerance: int = (
        np.inf
    )  # no constraint
    opt_roving_reference_confirmation_time_left_side_max_tolerance: int = (
        0  # no tolerance
    )
    # PIRA/RAW options - ignored if no relapses specified
    opt_raw_before_relapse_max_days: int = 30
    opt_raw_after_relapse_max_days: int = 90
    opt_pira_allow_relapses_between_event_and_confirmation: bool = False
    # Minimum increase options
    opt_max_score_that_requires_plus_1: float = 5.0
    opt_larger_increment_from_0: bool = True
    # Confirmation options
    opt_require_confirmation: bool = True
    opt_confirmation_time: float = 6 * 30  # > 0, or -1 for sustained over follow-up
    opt_confirmation_type: str = "minimum"  # "minimum" or "monotonic"
    opt_confirmation_included_values: str = "all"  # "last" or "all"
    opt_confirmation_sustained_minimal_distance: int = 0  # only if "sustained"
    opt_confirmation_time_right_side_max_tolerance: int = np.inf  # not for "sustained"
    opt_confirmation_time_left_side_max_tolerance: int = 0  # not for "sustained"
    opt_confirmation_require_confirmation_for_last_visit: bool = (
        True  # If False, the last assessment doesn't need confirmation
    )
    # Minimal distance options
    opt_minimal_distance_time: int = 0
    opt_minimal_distance_type: str = "reference"  # "reference" or "previous"
    opt_minimal_distance_backtrack_decrease: bool = (
        True  # go back to last low enough reference
    )
    # Input specifications
    edss_score_column_name: str = "edss_score"
    time_column_name: str = "days_after_baseline"
    # Output specifications
    time_since_last_relapse_column_name: str = "days_since_previous_relapse"
    time_to_next_relapse_column_name: str = "days_to_next_relapse"
    is_general_rebaseline_flag_column_name: str = "is_general_rebaseline"
    is_raw_pira_rebaseline_flag_column_name: str = "is_raw_pira_rebaseline"
    is_post_relapse_rebaseline_flag_column_name: str = "is_post_relapse_rebaseline"
    is_post_event_rebaseline_flag_column_name: str = "is_post_event_rebaseline"
    used_as_general_reference_score_flag_column_name: str = (
        "edss_score_used_as_new_general_reference"
    )
    used_as_raw_pira_reference_score_flag_column_name: str = (
        "edss_score_used_as_new_raw_pira_reference"
    )
    is_progression_flag_column_name: str = "is_progression"
    progression_type_column_name: str = "progression_type"
    progression_score_column_name: str = "progression_score"
    progression_reference_score_column_name: str = "progression_reference_score"
    progression_event_id_column_name: str = "progression_event_id"
    label_undefined_progression: str = "Undefined"
    label_pira: str = "PIRA"
    label_pira_confirmed_in_raw_window: str = "PIRA confirmed in RAW window"
    label_raw: str = "RAW"

    def __post_init__(self):
        """Non-boilerplate __init__ part."""
        # --------------------------------------------------------------------------
        # Check argument values
        # --------------------------------------------------------------------------
        if (not self.return_first_event_only) and (
            self.undefined_progression
            not in [
                "re-baselining only",
                "never",
                "all",
                "end",
            ]
        ):
            raise ValueError(
                "Invalid undefined progression option! Available options: 're-baselining only', 'never', 'all', 'end'."
            )
        if self.return_first_event_only:
            if self.undefined_progression == "end":
                raise ValueError(
                    "Undefined progression option 'end' can not be used if 'return_first_event_only' is true."
                )
            if self.undefined_progression not in [
                "re-baselining only",
                "never",
                "all",
            ]:
                raise ValueError(
                    "Invalid undefined progression option! Available options: 're-baselining only', 'never', 'all'."
                )
        if self.undefined_progression_wrt_raw_pira_baseline not in [
            "greater only",
            "equal or greater",
            "any",
        ]:
            raise ValueError(
                "Invalid option for undefined progression w.r.t. RAW/PIRA baseline. Available options: 'greater only', 'equal or greater', 'any'."
            )
        if self.merge_continuous_events and (
            self.continuous_events_max_repetition_time < 0
        ):
            raise ValueError("Max. repetition time for merging events must be >= 0.")
        # Baseline arguments
        if self.opt_baseline_type not in [
            "fixed",
            "roving",
        ]:
            raise ValueError(
                "Invalid baseline option! Available options: 'fixed', 'roving'."
            )
        if (self.opt_roving_reference_require_confirmation) and (
            self.opt_roving_reference_confirmation_time <= 0
        ):
            raise ValueError(
                "Invalid input for confirmation interval. If confirmation of roving reference required, choose a duration > 0."
            )
        if self.opt_roving_reference_confirmation_included_values not in [
            "last",
            "all",
        ]:
            raise ValueError(
                "Invalid option for roving reference confirmation values! Available options: 'all', 'last'."
            )
        if self.opt_roving_reference_confirmation_time_right_side_max_tolerance < 0:
            raise ValueError(
                "Roving reference confirmation right tolerance must be >= 0."
            )
        if self.opt_roving_reference_confirmation_time_left_side_max_tolerance < 0:
            raise ValueError(
                "Roving reference confirmation left tolerance must be >= 0."
            )
        # RAW/PIRA arguments
        if self.opt_raw_before_relapse_max_days < 0:
            raise ValueError("Max. RAW time before relapse must be >= 0.")
        if self.opt_raw_after_relapse_max_days < 0:
            raise ValueError("Max. RAW time after relapse must be >= 0.")
        # Confirmation arguments
        if self.opt_require_confirmation:
            if (self.opt_confirmation_time != -1) and (self.opt_confirmation_time <= 0):
                raise ValueError(
                    "Invalid input for confirmation interval. If confirmation required, choose -1 for sustained or a duration > 0."
                )
            if (self.opt_confirmation_time == -1) and (
                self.opt_confirmation_included_values == "last"
            ):
                raise ValueError(
                    "Invalid confirmation requirements. For sustained progession, only the option 'all' is valid for included values."
                )
        if self.opt_confirmation_type not in [
            "minimum",
            "monotonic",
        ]:
            raise ValueError(
                "Invalid confirmation criterion type. Options are 'minimum' or 'monotonic'."
            )
        if self.opt_confirmation_included_values not in [
            "all",
            "last",
        ]:
            raise ValueError(
                "Invalid confirmation scores type. Options are 'all' or 'last'."
            )
        if self.opt_confirmation_sustained_minimal_distance < 0:
            raise ValueError("Minimal distance for sustained must be >= 0.")
        if self.opt_confirmation_time_right_side_max_tolerance < 0:
            raise ValueError("Confirmation right hand side tolerance must be >= 0.")
        if self.opt_confirmation_time_left_side_max_tolerance < 0:
            raise ValueError("Confirmation left hand side tolerance must be >= 0.")
        # Minimal distance arguments
        if self.opt_minimal_distance_type not in [
            "reference",
            "previous",
        ]:
            raise ValueError(
                "Invalid minimal distance type. Options are 'reference' or 'previous'."
            )
        if self.opt_minimal_distance_time < 0:
            raise ValueError("Invalid minimal distance time, must be >= 0.")

        # --------------------------------------------------------------------------
        # Set additional variables
        # --------------------------------------------------------------------------
        self.baseline_score_column_name = "baseline_score"
        self.baseline_timestamp_column_name = "baseline_timestamp"
        # For the 'end' option for undefined progression, we first have to annotate
        # events using the 're-baselining only' option, then a second time using the
        # 'all' option for the assessments after the last event. Therefore, we define
        # a new variable to pass to the first round of annotation.
        if self.undefined_progression == "end":
            self.undefined_progression_to_pass_on = "re-baselining only"
        else:
            self.undefined_progression_to_pass_on = self.undefined_progression

    def _is_above_progress_threshold(
        self,
        current_edss,
        reference_edss,
    ):
        """Determine if a score meets the minimum increase condition.

        Standard definition (Müller 2023):
        - baseline EDSS 0.0: increase of at least 1.5
        - baseline EDSS > 0.0 and <= 5.0: increase of at least 1.0
        - baseline EDSS >= 5.5: increase of at least 0.5

        For a minimal increase of 0.5 irrespective of baseline, choose
        opt_larger_increment_from_0=False and opt_max_score_that_requires_plus_1=-1.
        For a minimal increase of 1.0 irrespective of baseline, choose
        opt_larger_increment_from_0=False and opt_max_score_that_requires_plus_1=10.0.

        Args:
        - current_edss: an EDSS score
        - reference_edss: the reference to which current_edss is compared

        Returns:
        - bool: True if above progress threshold

        """
        if self.opt_larger_increment_from_0 and reference_edss == 0:
            minimal_increase = 1.5
        else:
            if reference_edss <= self.opt_max_score_that_requires_plus_1:
                minimal_increase = 1
            else:
                minimal_increase = 0.5
        if current_edss >= reference_edss + minimal_increase:
            return True
        else:
            return False

    def _get_confirmation_scores_dataframe(
        self,
        current_timestamp,
        follow_up_dataframe,
        opt_confirmation_time,
        opt_confirmation_included_values,
        opt_confirmation_sustained_minimal_distance,
        opt_confirmation_time_right_side_max_tolerance,
        opt_confirmation_time_left_side_max_tolerance,
    ):
        """TBD, some thoughts:

        The function to get the confirmation scores is separated from
        the function that actually checks the confirmation condition.
        This will be useful for assessing RAW and PIRA, where we have
        to check for relapses within the confirmation period or in
        proximity of the confirmation score.

        We will use the same function to get the confirmation scores
        for the roving reference. We could use it for post-relapse
        re-baselining, too, but this is not yet implemented. This is
        also the reason why we pass the confirmation options as args
        and not via 'self'.

        Implementation notes
        -   By default, the confirmation interval is unbounded to the
            right, i.e. if confirmation is required at 12 weeks, the
            first assessment >= 12 weeks from the event is considered
            as confirmation assessment irrespective of its distance.
            This can be restricted usint the right side max tolerance
            argument (default is infinite) such that events that are
            after confirmation time plus this tolerance will not be
            considered confirmation assessments.
        -   The argument for left hand tolerance is technically not
            needed, because it simply sets the confirmation time to
            confirmation time - tolerance.
        -   By default, there is no minimum duration of post-event
            follow-up required for 'sustained'. Such a minimal distance
            can be set via the sustained minimal distance argument.

        """
        assessments_after_event_candidate = follow_up_dataframe[
            follow_up_dataframe[self.time_column_name] > current_timestamp
        ]
        # If sustained, just take all that are compatible with the minimal
        # distance condition (which is 0 by default).
        # NOTE: The '>=' is required here because the minimal distance is
        # measured from the event candidate; if the distance is 0, an event
        # can anyways not confirm itself due to the '>' in the assignment
        # above, so this is safe.
        if opt_confirmation_time == -1:
            confirmation_scores_dataframe = assessments_after_event_candidate[
                assessments_after_event_candidate[self.time_column_name]
                >= current_timestamp + opt_confirmation_sustained_minimal_distance
            ]
        # If not, start slicing... Idea: take all assessments >= x after,
        # then obtain the index of the first entry, then for confirmation
        # take all rows from current up to and including this index.
        # NOTE: For next confirmation, choose a tiny interval such as 0.5,
        # don't allow tolerance to the left, and leave the right side
        # unbounded.
        else:
            # Check if the constraint for the maximal distance between
            # an event candidate and the confirmation assessment is met.
            assessments_after_end_of_confirmation_interval = (
                assessments_after_event_candidate[
                    (
                        assessments_after_event_candidate[self.time_column_name]
                        >= current_timestamp
                        + opt_confirmation_time
                        - opt_confirmation_time_left_side_max_tolerance
                    )
                    & (
                        assessments_after_event_candidate[self.time_column_name]
                        <= current_timestamp
                        + opt_confirmation_time
                        + opt_confirmation_time_right_side_max_tolerance
                    )
                ].copy()
            )
            # If there are no confirmation scores available, just return
            # an empty dataframe. This if/else is required because the
            # slicing in the 'else' part would throw an error if we used
            # it on an empty dataframe.
            if len(assessments_after_end_of_confirmation_interval) == 0:
                confirmation_scores_dataframe = (
                    assessments_after_end_of_confirmation_interval
                )
            else:
                first_index_after_confirmation_interval = (
                    assessments_after_end_of_confirmation_interval.iloc[0].name
                )
                # NOTE: loc includes the boundary, so the following takes all
                # values up to and including the index of the first at or after
                # confirmation time. See e.g. https://stackoverflow.com/a/31593712
                # for an explanation of the loc and iloc behaviours.
                confirmation_scores_dataframe = assessments_after_event_candidate.loc[
                    :first_index_after_confirmation_interval
                ]
                # If we only take the last score for confirmation, return
                # it as a one-row dataframe (not a series!)
                if opt_confirmation_included_values == "last":
                    confirmation_scores_dataframe = confirmation_scores_dataframe.iloc[
                        [-1]
                    ]

        return confirmation_scores_dataframe

    def _check_confirmation_scores_and_get_confirmed_score(
        self,
        current_edss,
        current_reference,
        confirmation_scores_dataframe,
        additional_lower_threshold,
    ):
        """TBD, some thoughts:

        Look at confirmatiom scores and check if they satisfy
        the confirmation conditions (minimal required increase,
        minimum or monotonic) with respect to the specified
        reference score.

        The confirmation type is loaded from self, since this
        function is only used for confirming events, not for
        confirming baselines.

        There is an optional argument additional_lower_threshold,
        which can be used to set the confirmation threshold to a
        given minimum value. This is used for undefined progression
        with a score constraint w.r.t. the RAW/PIRA baseline.

        """
        confirmed_flag = False
        confirmed_edss = np.nan
        # Make this function safe for empty confirmation dataframes.
        # We will check this before calling this function, but just in case...
        if len(confirmation_scores_dataframe) > 0:
            confirmation_scores = np.array(
                confirmation_scores_dataframe[self.edss_score_column_name]
            )
            if self.opt_confirmation_type == "minimum":
                if self._is_above_progress_threshold(
                    current_edss=min(confirmation_scores),
                    reference_edss=current_reference,
                ) and (min(confirmation_scores) >= additional_lower_threshold):
                    confirmed_flag = True
                    confirmed_edss = min(current_edss, min(confirmation_scores))
            elif self.opt_confirmation_type == "monotonic":
                if (min(confirmation_scores) >= current_edss) and (
                    min(confirmation_scores) >= additional_lower_threshold
                ):
                    confirmed_flag = True
                    confirmed_edss = current_edss

        return confirmed_flag, confirmed_edss

    def _backtrack_minimal_distance_compatible_reference(
        self,
        current_edss,
        current_timestamp,
        baselines_df,
    ):
        """TBD, some thoughts:

        The idea of this function is that in some cases an increase
        is preceded by a decrease which leads to a re-baselining too
        close to the potential increase. In this case, it would not
        be a progression, however it would have been if the baseline
        had been stable at a higher level. This might not really
        make sense from a clinical point of view. One solution would
        of course be to require confirmation of a new roving baseline
        over a time >= the minimal distance, but there are published
        works where the roving baseline is NOT confirmed, and even
        this would not solve the issue for all confirmation options.

        References before previous events are NOT allowed if we reset
        the baseline after an event. The same  holds for post-relapse
        re-baselining assessments. However, this can be ensured by
        selecting the 'baselines_df' input appropriately, thus we don't
        have to implement it here.

        The reference only ever decreases (overall in relapse-free,
        or per post-relapse or post-event period), so it makes sense
        to take the closest reference to the candidate, because the
        reference is the lowest possible this way. This is important
        for the confirmation step, where scores need to be larger
        than the reference and some increment...
        """

        # From all the previous references, flag those that are
        # low enough so that 'current_edss' would be a progression
        # with respect to them.
        previous_rebaselines = baselines_df.copy()
        previous_rebaselines["low_enough_for_progression"] = previous_rebaselines.apply(
            lambda row: self._is_above_progress_threshold(
                current_edss=current_edss,
                reference_edss=row[self.baseline_score_column_name],
            ),
            axis=1,
        )
        # Discard the ones that are not low enough and proceed
        # with the rest.
        previous_rebaselines = previous_rebaselines[
            previous_rebaselines["low_enough_for_progression"]
        ].copy()
        # If none of the previous references is low enough, we're
        # done...
        if len(previous_rebaselines) == 0:
            return np.nan, np.nan
        # ... else we have to get those that also fulfill the
        # minimal distance requirement.
        else:
            low_enough_and_far_enough = previous_rebaselines[
                previous_rebaselines[self.baseline_timestamp_column_name]
                + self.opt_minimal_distance_time
                <= current_timestamp
            ].copy()
            if len(low_enough_and_far_enough) > 0:
                return (
                    low_enough_and_far_enough.iloc[-1][self.baseline_score_column_name],
                    low_enough_and_far_enough.iloc[-1][
                        self.baseline_timestamp_column_name
                    ],
                )
            else:
                return np.nan, np.nan

    def _add_relapses_to_follow_up(
        self,
        follow_up_df,
        relapse_timestamps,
    ):
        """TBD, some thoughts:

        The relapses are provided via a list of timestamps. From
        this list, we compute for each EDSS assessment the time
        since the last relapse and the time to the next relapse.

        """
        # Work on a copy to avoid manipulating the original data.
        follow_up_df_with_relapses = follow_up_df.copy()
        # If no relapses are provided, we're done.
        if relapse_timestamps == []:
            follow_up_df_with_relapses[self.time_since_last_relapse_column_name] = (
                np.nan
            )
            follow_up_df_with_relapses[self.time_to_next_relapse_column_name] = np.nan
        else:
            # Make sure the relapse timestamps are well ordered.
            relapse_timestamps = sorted(relapse_timestamps)
            # Merge the largest relapse timestamp before an
            # assessment to each assessment (the last relapse
            # before the assessment). May be on the day of
            # the assessment itself.
            follow_up_df_with_relapses = pd.merge_asof(
                left=follow_up_df_with_relapses,
                right=pd.DataFrame(
                    {
                        "previous_relapse_" + self.time_column_name: relapse_timestamps,
                    }
                ),
                left_on=self.time_column_name,
                right_on="previous_relapse_" + self.time_column_name,
                allow_exact_matches=True,
                direction="backward",
            )
            # Merge the smallest relapse timestamp after an
            # assessment to each assessment (the first relapse
            # after the assessment). May be on the day of
            # the assessment itself.
            follow_up_df_with_relapses = pd.merge_asof(
                left=follow_up_df_with_relapses,
                right=pd.DataFrame(
                    {
                        "next_relapse_" + self.time_column_name: relapse_timestamps,
                    }
                ),
                left_on=self.time_column_name,
                right_on="next_relapse_" + self.time_column_name,
                allow_exact_matches=True,
                direction="forward",
            )
            # Compute the timedeltas between relapse and assessment.
            follow_up_df_with_relapses[self.time_since_last_relapse_column_name] = (
                follow_up_df_with_relapses[self.time_column_name]
                - follow_up_df_with_relapses[
                    "previous_relapse_" + self.time_column_name
                ]
            )
            follow_up_df_with_relapses[self.time_to_next_relapse_column_name] = (
                follow_up_df_with_relapses["next_relapse_" + self.time_column_name]
                - follow_up_df_with_relapses[self.time_column_name]
            )
            # Drop the previous/next relapse timestamp.
            follow_up_df_with_relapses = follow_up_df_with_relapses.drop(
                columns=[
                    "previous_relapse_" + self.time_column_name,
                    "next_relapse_" + self.time_column_name,
                ]
            )
            # Unfortunately, the merges drop the index, so we have to do a
            # little workaround to recover them. Dropping the index makes
            # sense technically, because there's no guarantee that the merge
            # is 1:1, but here we're fine.
            # We keep the index because we will need it later for correctly
            # assigning values to rows.
            merge_scaffold = follow_up_df[[self.time_column_name]].copy()
            follow_up_df_with_relapses = (
                merge_scaffold.reset_index()
                .merge(follow_up_df_with_relapses, on=self.time_column_name, how="left")
                .set_index("index")
                .rename_axis(None)
            )

        return follow_up_df_with_relapses

    def _get_post_relapse_rebaseline_timestamps(
        self,
        follow_up_df,
        relapse_timestamps,
    ):
        """TBD, some thoughts:

        We need a function that returns the re-baselining assessment
        timestamps for each relapse. We need to jump through a couple
        of hoops here to correctly account for overlapping RAW windows
        and sequences of relapses with no assessments in between.

        """
        # Helper column, will be dropped.
        relapse_timestamp_column_name = "relapse_timestamp"
        relapses = pd.DataFrame({relapse_timestamp_column_name: relapse_timestamps})
        rebaselines = []
        # If there are no relapses, we're done.
        if len(relapses) == 0:
            pass
        else:
            # Add buffer times around the relapses (the RAW-window)
            relapses["start_buffer"] = (
                relapses[relapse_timestamp_column_name]
                - self.opt_raw_before_relapse_max_days
            )
            relapses["stop_buffer"] = (
                relapses[relapse_timestamp_column_name]
                + self.opt_raw_after_relapse_max_days
            )
            # Add the start time of the next relapse to each line
            relapses["start_next"] = relapses["start_buffer"].shift(-1)
            # Now find the rebaseline assessment for each relapse
            for _, row in relapses.iterrows():
                rebaseline_candidates = follow_up_df[
                    follow_up_df[self.time_column_name]
                    > row[relapse_timestamp_column_name]
                    + self.opt_raw_after_relapse_max_days
                ]
                # If there are candidates, take the first one
                if len(rebaseline_candidates) > 0:
                    rebaseline = rebaseline_candidates.iloc[0]
                    # Check if it is within an overlapping next relapse.
                    if row["stop_buffer"] >= row["start_next"]:
                        pass
                    else:
                        rebaselines = rebaselines + [rebaseline[self.time_column_name]]

        # NOTE: There might be duplicates, so drop them.
        rebaselines = list(dict.fromkeys(rebaselines))
        # This list is empty if no relapses are present.
        return rebaselines

    def _check_assessment_for_progression(
        self,
        check_raw_pira,
        annotated_df,
        relapse_timestamps,
        baselines_df,
        current_assessment_index,
        additional_lower_threshold,
    ):
        """TBD, some thoughts:

        This function check if an EDSS score is an event by
        checking the minimal distance, minimal increase, and
        confirmation conditions. It also checks whether an
        event is RAW, PIRA, PIRA confirmed within RAW window,
        or undefined progression.

        Returns progression yes/no, type, event score, and
        the reference score for the event.

        """
        row = annotated_df.loc[current_assessment_index]
        current_edss = row[self.edss_score_column_name]
        current_timestamp = row[self.time_column_name]

        is_progression = False
        progression_type = None
        confirmed_event_score = np.nan

        # If the score is below our additional lower threshold, it is not a
        # progression candidate anyways.
        if current_edss >= additional_lower_threshold:
            # The minimal distance has to be checked first, since it
            # can change the reference score if we allow backtracking.
            minimal_distance_condition_satisfied = True
            current_baseline_score = baselines_df.iloc[-1]["baseline_score"]
            if self.opt_minimal_distance_time > 0:
                if self.opt_minimal_distance_type == "previous":
                    previous_timestamp = annotated_df.loc[current_assessment_index - 1][
                        self.time_column_name
                    ]
                    distance = current_timestamp - previous_timestamp
                elif self.opt_minimal_distance_type == "reference":
                    distance = (
                        current_timestamp - baselines_df.iloc[-1]["baseline_timestamp"]
                    )
                    if self.opt_minimal_distance_backtrack_decrease:
                        (
                            backtracked_reference,
                            backtracked_timestamp,
                        ) = self._backtrack_minimal_distance_compatible_reference(
                            current_edss=current_edss,
                            current_timestamp=current_timestamp,
                            baselines_df=baselines_df,
                        )
                        if backtracked_timestamp >= 0:
                            distance = current_timestamp - backtracked_timestamp
                            current_baseline_score = backtracked_reference

                if distance < self.opt_minimal_distance_time:
                    minimal_distance_condition_satisfied = False

            # Now that the distance is checked, check if the increase is large enough.
            if minimal_distance_condition_satisfied:
                # Does it qualify as progression?
                is_progression_candidate = self._is_above_progress_threshold(
                    current_edss=current_edss,
                    reference_edss=current_baseline_score,
                )
                if is_progression_candidate:
                    # Determine the progression type. NOTE: this might then still
                    # change during confirmation for RAW and PIRA.
                    if check_raw_pira:
                        # Assume it's PIRA, and switch to RAW once any PIRA
                        # condition is violated.
                        progression_type = self.label_pira
                        if (
                            row[self.time_to_next_relapse_column_name]
                            <= self.opt_raw_before_relapse_max_days
                        ) or (
                            row[self.time_since_last_relapse_column_name]
                            <= self.opt_raw_after_relapse_max_days
                        ):
                            progression_type = self.label_raw
                    else:
                        progression_type = self.label_undefined_progression

                    # If we don't require confirmation, we're done.
                    if not self.opt_require_confirmation:
                        # If the event is within proximity of a relapse,
                        # it's RAW, else it's PIRA or "Undefined".
                        is_progression = True
                        confirmed_event_score = current_edss
                    # If the last assessment is exempt from confirmation,
                    # we can also skip the confirmation step.
                    elif (
                        self.opt_require_confirmation
                        and (
                            not self.opt_confirmation_require_confirmation_for_last_visit
                        )
                        and (
                            current_timestamp
                            == annotated_df[self.time_column_name].max()
                        )
                    ):
                        is_progression = True
                        confirmed_event_score = current_edss
                    else:
                        # First, get the confirmation score dataframe.
                        confirmation_scores_dataframe = self._get_confirmation_scores_dataframe(
                            current_timestamp=current_timestamp,
                            follow_up_dataframe=annotated_df,
                            opt_confirmation_time=self.opt_confirmation_time,
                            opt_confirmation_included_values=self.opt_confirmation_included_values,
                            opt_confirmation_sustained_minimal_distance=self.opt_confirmation_sustained_minimal_distance,
                            opt_confirmation_time_right_side_max_tolerance=self.opt_confirmation_time_right_side_max_tolerance,
                            opt_confirmation_time_left_side_max_tolerance=self.opt_confirmation_time_left_side_max_tolerance,
                        )
                        # If we don't have any confirmation scores, we're done.
                        # Otherwise we now have to check the conditions.
                        if len(confirmation_scores_dataframe) > 0:
                            # Check if confirmed; if not, we don't even have to
                            # bother with the relapses...
                            (
                                is_progression,
                                confirmed_event_score,
                            ) = self._check_confirmation_scores_and_get_confirmed_score(
                                current_edss=current_edss,
                                current_reference=current_baseline_score,  # The UP vs. RAW/PIRA version choice happens at the start.
                                confirmation_scores_dataframe=confirmation_scores_dataframe,
                                additional_lower_threshold=additional_lower_threshold,
                            )
                            # If unconfirmed, nope, otherwise continue.
                            if is_progression:
                                # If we already know that it's RAW or UP, we're done.
                                # Otherwise we have to check for relapses during confirmation,
                                # unless there are no relapses, of course...
                                # We also introduce a new type 'PIRA confirmed during relapse'
                                # to mark events that are outside the RAW window, but confirmed
                                # by assessments during relapses. Treated like PIRA and RAW for
                                # baselines etc.
                                if (progression_type == self.label_pira) and (
                                    len(relapse_timestamps) > 0
                                ):
                                    # If we don't allow relapses between the event and the
                                    # end of the confirmation interval, or if we consider all
                                    # scores for confirmation, then any event with relapses
                                    # in this time interval is 'PIRA confirmed within RAW window'.
                                    # We check the relapse timestamp list for whether there are
                                    # any in between the assessments. We can use >= and <= since
                                    # if the assessment were at the same time as a relapse, it
                                    # would be RAW anyways...
                                    if (
                                        not self.opt_pira_allow_relapses_between_event_and_confirmation
                                    ) or (
                                        self.opt_confirmation_included_values == "all"
                                    ):
                                        last_confirmation_score_timestamp = (
                                            confirmation_scores_dataframe[
                                                self.time_column_name
                                            ].max()
                                        )
                                        # If any relapse timestamp >= current and <= last conf + buffer,
                                        # the event is 'PIRA confirmed within RAW window'.
                                        if max(
                                            (
                                                np.array(relapse_timestamps)
                                                >= current_timestamp
                                            )
                                            & (
                                                np.array(relapse_timestamps)
                                                <= last_confirmation_score_timestamp
                                                + self.opt_raw_before_relapse_max_days
                                            )
                                        ):
                                            progression_type = (
                                                self.label_pira_confirmed_in_raw_window
                                            )

                                    # We also have to make sure that there is no relapse
                                    # too close before or after the confirmation interval!
                                    # NOTE: This is covered by the part above if relapses
                                    # within confirmation interval are not allowed or all
                                    # values are relevant for confirmation. TODO: remove
                                    # this redundancy.
                                    if (
                                        confirmation_scores_dataframe.iloc[-1][
                                            self.time_to_next_relapse_column_name
                                        ]
                                        <= self.opt_raw_before_relapse_max_days
                                    ):
                                        progression_type = (
                                            self.label_pira_confirmed_in_raw_window
                                        )
                                    if (
                                        confirmation_scores_dataframe.iloc[0][
                                            self.time_since_last_relapse_column_name
                                        ]
                                        <= self.opt_raw_after_relapse_max_days
                                    ):
                                        progression_type = (
                                            self.label_pira_confirmed_in_raw_window
                                        )

        return (
            is_progression,
            progression_type,
            confirmed_event_score,
            current_baseline_score,
        )

    def _combine_events_forward_lookup(
        self,
        annotated_df,
        baselines_df,
        relapse_timestamps,
        iid_index,
        iid_confirmed_event_score,
        iid_progression_type,
        additional_lower_threshold,
    ):
        """TBD, some thoughts:

        This is to identify connected events; we only look at strictly
        monotonically increasing scores, with an optional tolerance for
        identical scores recorded in close temporal proximity.

        Notes:
        *   Just a little fluke improvement already stops this process...
            Show this quirk in the documentation!
        *   Assessments considered as repetition measurements (i.e.
            within continuous_events_max_repetition_time) are also flagged
            as members of the merged event, but not if they are at the end.
        *   This is meant to be used for PIRA/RAW; undefined events are
            always considered singular.
        *   Events included into a merged event series don't get their own
            'is progression' flag or a progression type/score/reference.
            This is by design in order to make analysis easier (e.g. event
            counts based on rows with 'is_progression == True'). They can
            be identified via the event ID.
        """
        # Setup loop... We collect the indices of each assessment that
        # is part of the loop in a list, and we also keep track of potential
        # stabilizations or improvements.
        indices_of_merged_event = [iid_index]
        stagnation_started = False
        stagnation_timestamp = annotated_df.at[iid_index, self.time_column_name]
        last_confirmed_progression_timestamp = annotated_df.at[
            iid_index, self.time_column_name
        ]
        # Now we check each subsequent assessment until we find a
        # stabilization or improvement. We also initialize a list
        # where we collect indices of stagnation events, so if they
        # turn out to be at the end of a merge we can drop them.
        confirmed_event_score = iid_confirmed_event_score
        ids_final_stagnation_to_remove = []
        for i, row in annotated_df.loc[iid_index + 1 :].iterrows():
            # If the assessment is past the maximal allowed merge
            # distance, we stop.
            if (
                row[self.time_column_name]
                > last_confirmed_progression_timestamp
                + self.continuous_events_max_merge_distance
            ):
                break
            # If the score is lower than the current confirmed event
            # score, we stop. In this case, any confirmed score would
            # be lower than the previous one anyways.
            if row[self.edss_score_column_name] < confirmed_event_score:
                break
            # Else we need to test whether the next score from the
            # next assessment would be a progression itself. We use
            # the same baseline as we used for the IID.
            else:
                (
                    new_is_progression,
                    new_progression_type,
                    new_confirmed_event_score,
                    _,
                ) = self._check_assessment_for_progression(
                    check_raw_pira=True,
                    annotated_df=annotated_df,
                    relapse_timestamps=relapse_timestamps,
                    baselines_df=baselines_df,
                    current_assessment_index=i,
                    additional_lower_threshold=additional_lower_threshold,
                )
                # If the new score is not a progression w.r.t. the IID
                # baseline anymore, we stop the merge. This could happen
                # if e.g. a 'next confirmed' requirement is in place.
                if not new_is_progression:
                    break
                # We also have to check whether the progression is
                # still of the same type; otherwise we also stop.
                if new_progression_type != iid_progression_type:
                    break
                # If the confirmed score is lower than the previous
                # one, we consider the merged event over.
                if new_confirmed_event_score < confirmed_event_score:
                    break
                else:
                    # If the new score leads to an increased event score,
                    # we reset the stagnation flag and clear the IDs of
                    # stagnation events, since they are now not at the end
                    # of the merge anymore.
                    if new_confirmed_event_score > confirmed_event_score:
                        # Reset the stagnation flag
                        stagnation_started = False
                        # Also reset the IDs to remove list
                        ids_final_stagnation_to_remove = []
                    # If we observe a stagnation with respect to the confirmed event score,
                    # we check whether this event is close enough to the start of the
                    # stabilization period to be considered a repetition of measurement
                    # instead of a confirmation of stabilization.
                    elif new_confirmed_event_score == confirmed_event_score:
                        # If it is the first score in a series of stable scores, we keep
                        # the stabilization initiation timestamp and set the 'stabilization
                        # started' flag.
                        if not stagnation_started:
                            stagnation_started = True
                            # It started at the previous step, so we take the timestamp from there.
                            stagnation_timestamp = annotated_df.loc[i - 1][
                                self.time_column_name
                            ]
                        # If the current score is close enough to the previous one, we continue
                        # our loop, but keep track of the index.
                        if (
                            row[self.time_column_name] - stagnation_timestamp
                            <= self.continuous_events_max_repetition_time
                        ):
                            # We keep track of the IDs for the stabilization events; if
                            # they turn out to be at the end, we don't include them in
                            # the merged event.
                            ids_final_stagnation_to_remove = (
                                ids_final_stagnation_to_remove + [i]
                            )
                        # If it is past this tolerance window, we consider it a stabilization
                        # and consider the merged event over.
                        else:
                            break

                    # Continue the loop with this new score
                    confirmed_event_score = new_confirmed_event_score
                    last_confirmed_progression_timestamp = row[self.time_column_name]
                    indices_of_merged_event = indices_of_merged_event + [i]

        # Remove final stagnation
        indices_of_merged_event = [
            idx
            for idx in indices_of_merged_event
            if idx not in ids_final_stagnation_to_remove
        ]

        return (
            indices_of_merged_event,
            confirmed_event_score,
            last_confirmed_progression_timestamp,
        )

    def _annotate_progression_events(
        self,
        follow_up_dataframe,
        relapse_timestamps,
        undefined_progression,
    ):
        """TBD"""

        # Prepare the return dataframe
        annotated_df = follow_up_dataframe.copy()
        # Add relapse timedeltas; if no relapses are provided,
        # it will just generate NaNs.
        annotated_df = self._add_relapses_to_follow_up(
            follow_up_df=annotated_df,
            relapse_timestamps=relapse_timestamps,
        )

        # Let's keep track of the baselines for easier debugging...
        # The following flag marks post-event re-baselining events.
        annotated_df[self.is_post_event_rebaseline_flag_column_name] = False
        # The following flags mark all assessments where the
        # RAW/PIRA or the general baseline is reset. This always
        # coincides with post-event re-baselining for both, and
        # with post-relapse re-baselining for RAW/PIRA. If we use
        # a roving reference, this also flags all assessment where
        # a new roving reference is set.
        annotated_df[self.is_general_rebaseline_flag_column_name] = False
        annotated_df[self.is_raw_pira_rebaseline_flag_column_name] = False
        # Let's also keep track of the scores that are actually
        # carried forward after a re-baselining (in case of event
        # or baseline confirmation constraints, the new baseline is
        # not equivalent to the EDSS score determined at the assessment...)
        annotated_df[self.used_as_general_reference_score_flag_column_name] = np.nan
        annotated_df[self.used_as_raw_pira_reference_score_flag_column_name] = np.nan
        # Also initialize columns for progression annotation. We keep
        # track of the event, event type, event score, event reference
        # score, and event ID.
        annotated_df[self.is_progression_flag_column_name] = False
        annotated_df[self.progression_type_column_name] = None
        annotated_df[self.progression_score_column_name] = np.nan
        annotated_df[self.progression_reference_score_column_name] = np.nan
        annotated_df[self.progression_event_id_column_name] = np.nan
        # Initialize the confirmed event ID. Don't add this to the annotated
        # dataframe, we don't want to give any ID to non-events. The ID is
        # set to 0 here and then incremented by + 1 at each event, such that
        # the first event will start with 1.
        progression_event_id = 0

        # Get the study baseline and timestamp, prepare baseline dataframes.
        # We will then append new baselines if they are updated (roving), or
        # discard previous ones in case of an event (general and RAW/PIRA) or
        # a post-relapse re-baselining (RAW/PIRA only).
        # The very first row serves as baseline. The indices are not reset,
        # so we use iloc, not loc.
        study_baseline_score = annotated_df.iloc[0][self.edss_score_column_name]
        study_baseline_timestamp = annotated_df.iloc[0][self.time_column_name]
        general_baselines = pd.DataFrame(
            {
                "baseline_score": [study_baseline_score],
                "baseline_timestamp": [study_baseline_timestamp],
            }
        )
        raw_pira_baselines = pd.DataFrame(
            {
                "baseline_score": [study_baseline_score],
                "baseline_timestamp": [study_baseline_timestamp],
            }
        )

        # Flag all post-relapse re-baselining assessments. We will
        # later use these flags to identify assessments where we have
        # to check for undefined progression, since they don't qualify
        # for RAW or PIRA. We also need this flag in order to identify
        # assessments with residual post-relapse disability that set
        # a new baseline for RAW and PIRA.
        # Initialize the column and just set relapse assessments to True
        # via a list of relapse timestamps.
        annotated_df[self.is_post_relapse_rebaseline_flag_column_name] = False
        rebaselines_list = self._get_post_relapse_rebaseline_timestamps(
            follow_up_df=annotated_df,
            relapse_timestamps=relapse_timestamps,
        )
        annotated_df.loc[
            annotated_df["days_after_baseline"].isin(rebaselines_list),
            self.is_post_relapse_rebaseline_flag_column_name,
        ] = True

        # If we merge events, we keep track of indices we want to skip. This
        # is because the event merging happens inside the loop that checks
        # each assessment for progression, thus assessments that are already
        # included in a merged event must be skipped by the loop.
        indices_to_skip = []

        # Now start looping over the follow up, skipping the first row because
        # the first row can never be an event anyways.
        # NOTE: iterrows returns row indices, not row positions. See e.g.
        # https://stackoverflow.com/a/31593712 for a thorough explanation
        # of the differences between loc and iloc.
        for i, row in annotated_df.iloc[1:].iterrows():
            if i not in indices_to_skip:
                current_edss = row[self.edss_score_column_name]
                current_timestamp = row[self.time_column_name]

                is_progression = False
                progression_type = None
                confirmed_event_score = np.nan

                # Before we start, reset the additional_lower_threshold argument
                # which is used for certain options for undefined progression.
                additional_lower_threshold_for_progression_and_confirmation = 0

                # Now, check if the row is a post-relapse-rebaseline. Depending
                # on the outcome, we check for undefined progression or RAW/PIRA.
                # The idea here is to select the correct set of baselines.
                is_rebaseline_assessment = row[
                    self.is_post_relapse_rebaseline_flag_column_name
                ]
                # We also check for residual disability here.
                is_post_relapse_rebaselining_assessment_with_residual_disability = False
                if is_rebaseline_assessment:
                    if current_edss > raw_pira_baselines.iloc[-1]["baseline_score"]:
                        is_post_relapse_rebaselining_assessment_with_residual_disability = (
                            True
                        )

                # We need an additional switch to handle potential undefined
                # progression events with an event score equal to or lower than
                # the RAW/PIRA baseline. This constraint will also be applied to
                # confirmation scores.
                score_is_high_enough_for_undefined_wrt_raw_pira_baseline = False
                # If we allow undefined progression at assessments where the score
                # is equal or lower than the previous RAW/PIRA baseline, we have to
                # check all assessments that are post-relapse re-baselining assessments,
                # irrespective of whether a re-baselining actually happens (residual
                # disability) or not.
                if self.undefined_progression_wrt_raw_pira_baseline == "any":
                    score_is_high_enough_for_undefined_wrt_raw_pira_baseline = True
                # If we only allow undefined progression at assessments where the
                # score is >= RAW/PIRA baseline, we only check the assessments which
                # are >= RAW/PIRA baseline.
                elif (
                    self.undefined_progression_wrt_raw_pira_baseline
                    == "equal or greater"
                ):
                    if current_edss >= raw_pira_baselines.iloc[-1]["baseline_score"]:
                        score_is_high_enough_for_undefined_wrt_raw_pira_baseline = True
                        # Confirmation scores must also satisfy this condition, thus
                        # we keep it as a constraint for progression.
                        additional_lower_threshold_for_progression_and_confirmation = (
                            raw_pira_baselines.iloc[-1]["baseline_score"]
                        )
                # If we only allow undefined progression at assessments where the
                # score is > RAW/PIRA baseline, we only check the assessments which
                # are re-baselining and are > RAW/PIRA baseline (residual disability).
                elif self.undefined_progression_wrt_raw_pira_baseline == "greater only":
                    if current_edss > raw_pira_baselines.iloc[-1]["baseline_score"]:
                        score_is_high_enough_for_undefined_wrt_raw_pira_baseline = True
                        # Confirmation scores must also satisfy this condition, thus
                        # we keep it as a constraint for progression. Since the threshold
                        # is implemented as >=, we have to add + 0.5 here.
                        additional_lower_threshold_for_progression_and_confirmation = (
                            raw_pira_baselines.iloc[-1]["baseline_score"] + 0.5
                        )

                # Whether we check for RAW/PIRA or for undefined progression also
                # depends on our choice for the 'undefined_progression' argument.
                # If default 're-baselining only', get baseline dataframe and
                # the check_raw_pira flag depending on whether it is a post-relapse
                # re-baselining assessment, then run 'check_progression' once.
                # If 'never', only check progression for non-post-relapse re-
                # baselining assessments.
                # If 'all' and not a post-relapse re-baselining, first run the
                # check_progression function with RAW/PIRA, and if no progression
                # was found run it again to check for undefined.
                # If 'all general only', we check all assessments for undefined
                # progression, and don't check for RAW/PIRA. This option is not
                # allowed as argument in 'add_progression_events_to_follow_up'
                # since it is only used as a helper for the 'end' option (which
                # requires a re-run of '_annotate_progression_events' and is thus
                # not implemented here at this level).
                if undefined_progression == "re-baselining only":
                    if is_rebaseline_assessment:
                        # Whether we check for undefined also depends on the optional
                        # score constraint (candidate vs. RAW/PIRA baseline).
                        # NOTE: technically we could drop this, since the function
                        # that checks progression also checks the candidate itself
                        # against the lower threshold which we set in case of the
                        # 'equal or greater' or 'greater only' options. However,
                        # this if/else spares us some effort :)
                        if score_is_high_enough_for_undefined_wrt_raw_pira_baseline:
                            baselines_to_check = [
                                {
                                    "baselines_df": general_baselines,
                                    "check_raw_pira": False,
                                }
                            ]
                        else:
                            baselines_to_check = []
                    else:
                        baselines_to_check = [
                            {"baselines_df": raw_pira_baselines, "check_raw_pira": True}
                        ]
                elif undefined_progression == "never":
                    if is_rebaseline_assessment:
                        baselines_to_check = []
                    else:
                        baselines_to_check = [
                            {"baselines_df": raw_pira_baselines, "check_raw_pira": True}
                        ]
                elif undefined_progression == "all":
                    if is_rebaseline_assessment:
                        if score_is_high_enough_for_undefined_wrt_raw_pira_baseline:
                            baselines_to_check = [
                                {
                                    "baselines_df": general_baselines,
                                    "check_raw_pira": False,
                                }
                            ]
                        else:
                            baselines_to_check = []
                    else:
                        # Always check RAW/PIRA first!
                        baselines_to_check = [
                            {
                                "baselines_df": raw_pira_baselines,
                                "check_raw_pira": True,
                            },
                        ]
                        # The score constraint for undefined still applies!
                        if score_is_high_enough_for_undefined_wrt_raw_pira_baseline:
                            baselines_to_check = baselines_to_check + [
                                {
                                    "baselines_df": general_baselines,
                                    "check_raw_pira": False,
                                },
                            ]
                elif undefined_progression == "all general only":
                    # Whether we check for undefined also depends on the optional
                    # score constraint (candidate vs. RAW/PIRA baseline).
                    if score_is_high_enough_for_undefined_wrt_raw_pira_baseline:
                        baselines_to_check = [
                            {
                                "baselines_df": general_baselines,
                                "check_raw_pira": False,
                            }
                        ]
                    else:
                        baselines_to_check = []

                # If we want to check for progression, we now run the corresponding function.
                # We loop over all options (to enable the 'all' option or undefined progression),
                # but break the loop once a progression is found. Thus, if we use the 'all'
                # option, we don't overwrite RAW/PIRA events.
                # NOTE: if we have a score condition for undefined events (e.g. they must be >=
                # the current RAW/PIRA baseline), this condition must also hold for the confiration
                # scores. The threshold is 0 or the last RAW/PIRA baseline, so it has no effect
                # on RAW/PIRA assessment, but potentially on undefined progression.
                for baseline_option in baselines_to_check:
                    (
                        is_progression,
                        progression_type,
                        confirmed_event_score,
                        current_baseline_score,
                    ) = self._check_assessment_for_progression(
                        check_raw_pira=baseline_option["check_raw_pira"],
                        annotated_df=annotated_df,
                        relapse_timestamps=relapse_timestamps,
                        baselines_df=baseline_option["baselines_df"],
                        current_assessment_index=i,
                        additional_lower_threshold=additional_lower_threshold_for_progression_and_confirmation,
                    )
                    if is_progression:
                        progression_event_id = progression_event_id + 1
                        break

                # If we merge continuous RAW/PIRA events: more to come?
                if self.merge_continuous_events:
                    if is_progression and (
                        progression_type
                        in [
                            self.label_raw,
                            self.label_pira,
                            self.label_pira_confirmed_in_raw_window,
                        ]
                    ):
                        (
                            indices_of_merged_event,
                            confirmed_event_score,
                            last_confirmed_timestamp,
                        ) = self._combine_events_forward_lookup(
                            annotated_df=annotated_df,
                            relapse_timestamps=relapse_timestamps,
                            baselines_df=raw_pira_baselines,
                            iid_index=i,
                            iid_confirmed_event_score=confirmed_event_score,
                            iid_progression_type=progression_type,
                            additional_lower_threshold=0,
                        )
                    elif is_progression and (
                        progression_type in [self.label_undefined_progression]
                    ):
                        indices_of_merged_event = [i]
                        last_confirmed_timestamp = current_timestamp

                # New baseline? It depends on whether we have found a confirmed event
                # and on whether we are using a roving reference.
                # If there's a progression, we discard all our previous references
                # and continue with the confirmed event score. This will e.g. make
                # checking for the minimal distance with backtracking easier.
                if is_progression:
                    # Annotate results...
                    annotated_df.at[i, self.is_progression_flag_column_name] = True
                    annotated_df.at[i, self.progression_type_column_name] = (
                        progression_type
                    )
                    annotated_df.at[i, self.progression_score_column_name] = (
                        confirmed_event_score
                    )
                    annotated_df.at[i, self.progression_reference_score_column_name] = (
                        current_baseline_score
                    )
                    annotated_df.at[i, self.progression_event_id_column_name] = (
                        progression_event_id
                    )
                    # If we merge events: label them.
                    if self.merge_continuous_events:
                        indices_to_skip = indices_to_skip + indices_of_merged_event
                        for event_index in indices_of_merged_event:
                            annotated_df.at[
                                event_index, self.progression_event_id_column_name
                            ] = progression_event_id

                    # If we only want the first event, we can stop here and we do
                    # not have to bother anymore about baselines...
                    if self.return_first_event_only:
                        break

                    # ... and update baselines. We discard all previous baselines,
                    # unless the event is undefined with a confirmed event score
                    # equal or smaller than the RAW/PIRA baseline.
                    annotated_df.at[
                        i, self.is_post_event_rebaseline_flag_column_name
                    ] = True
                    # Relapse-independent baseline - this one is reset after any
                    # event irrespective of the event type.
                    annotated_df.at[i, self.is_general_rebaseline_flag_column_name] = (
                        True
                    )
                    annotated_df.at[
                        i, self.used_as_general_reference_score_flag_column_name
                    ] = confirmed_event_score
                    general_baseline_timestamp = current_timestamp
                    if self.merge_continuous_events:
                        general_baseline_timestamp = last_confirmed_timestamp
                    general_baselines = pd.DataFrame(
                        {
                            "baseline_score": [confirmed_event_score],
                            "baseline_timestamp": [general_baseline_timestamp],
                        }
                    )
                    # RAW/PIRA baseline

                    # If the confirmed event score is lower than the previous RAW/PIRA
                    # baseline - this can happen with undefined progression and the
                    # 'any' option (no constraint w.r.t. RAW/PIRA baseline) - then we
                    # don't change the RAW/PIRA baseline.

                    # Case 1 - the progression happens at a post-relapse re-baselining
                    # assessment. Subcases: A) The EDSS score is > than the previous
                    # RAW/PIRA baseline -> we adjust the RAW/PIRA baseline. B) The
                    # score is <= the previous RAW/PIRA baseline -> we do not adjust
                    # the RAW/PIRA baseline.
                    adjust_raw_pira_baseline = False
                    if is_rebaseline_assessment:
                        if current_edss > raw_pira_baselines.iloc[-1]["baseline_score"]:
                            # If it is a re-baselining assessment, we have to be careful
                            # to use the residual disability as a new baseline, even if
                            # the confirmed event score is lower.
                            # NOTE: current_edss < confirmed can not happen, since
                            # confirmation only ever leads to a lower event score.
                            new_raw_pira_baseline = current_edss
                            adjust_raw_pira_baseline = True
                    # Case 2 - the progression does not happen at a post-relapse re-
                    # baselining assessment. Subcases: A) The confirmed event score
                    # is > than the previous RAW/PIRA baseline -> we adjust the RAW/PIRA
                    # baseline. B) The event score is <= the previous RAW/PIRA baseline ->
                    # we do not adjust the RAW/PIRA baseline.
                    else:
                        if (
                            confirmed_event_score
                            > raw_pira_baselines.iloc[-1]["baseline_score"]
                        ):
                            new_raw_pira_baseline = confirmed_event_score
                            adjust_raw_pira_baseline = True

                    if adjust_raw_pira_baseline:
                        annotated_df.at[
                            i, self.is_raw_pira_rebaseline_flag_column_name
                        ] = True
                        annotated_df.at[
                            i, self.used_as_raw_pira_reference_score_flag_column_name
                        ] = new_raw_pira_baseline
                        raw_pira_baseline_timestamp = current_timestamp
                        if self.merge_continuous_events:
                            raw_pira_baseline_timestamp = last_confirmed_timestamp
                        raw_pira_baselines = pd.DataFrame(
                            {
                                "baseline_score": [new_raw_pira_baseline],
                                "baseline_timestamp": [raw_pira_baseline_timestamp],
                            }
                        )
                    else:
                        annotated_df.at[
                            i, self.is_post_relapse_rebaseline_flag_column_name
                        ] = False

                else:
                    # If we have a post-relapse re-baselining assessment, we need to reset
                    # the baselines for RAW/PIRA. This is unconfirmed (in contrast to an
                    # optionally confirmed disability progression event).
                    # NOTE: Only re-baseline if there's residual disability, see Müller et
                    # al.: "The reference should also be reset if a relapse causes residual
                    # disability". This means that if we're better off after a relapse, the
                    # baseline can go down, too, but only if we use a roving reference, and
                    # then it also has to satisfy the corresponding confirmation condition.
                    if is_post_relapse_rebaselining_assessment_with_residual_disability:
                        annotated_df.at[
                            i, self.is_post_relapse_rebaseline_flag_column_name
                        ] = True
                        annotated_df.at[
                            i, self.is_raw_pira_rebaseline_flag_column_name
                        ] = True
                        annotated_df.at[
                            i, self.used_as_raw_pira_reference_score_flag_column_name
                        ] = current_edss
                        raw_pira_baselines = pd.DataFrame(
                            {
                                "baseline_score": [current_edss],
                                "baseline_timestamp": [current_timestamp],
                            }
                        )
                    # Remove the post-relapse re-baselining flag otherwise.
                    elif is_rebaseline_assessment and (
                        current_edss <= raw_pira_baselines.iloc[-1]["baseline_score"]
                    ):
                        annotated_df.at[
                            i, self.is_post_relapse_rebaseline_flag_column_name
                        ] = False

                    # If we have a roving baseline, the baselines could improve.
                    # TODO: Write a function for this to avoid all the copying...
                    if self.opt_baseline_type == "roving":
                        # If there's a candidate, and confirmation is required,
                        # we get the confirmation scores here because they are
                        # the same for both baselines.
                        general_roving_confirmed = True
                        raw_pira_roving_confirmed = True
                        if self.opt_roving_reference_require_confirmation and (
                            (
                                current_edss
                                < raw_pira_baselines.iloc[-1]["baseline_score"]
                            )
                            or (
                                current_edss
                                < general_baselines.iloc[-1]["baseline_score"]
                            )
                        ):
                            roving_rebaseline_confirmation_scores_df = self._get_confirmation_scores_dataframe(
                                current_timestamp=current_timestamp,
                                follow_up_dataframe=annotated_df,
                                opt_confirmation_time=self.opt_roving_reference_confirmation_time,
                                opt_confirmation_included_values=self.opt_roving_reference_confirmation_included_values,
                                opt_confirmation_sustained_minimal_distance=0,  # Sustained is a pointless option for the baseline anyways...
                                opt_confirmation_time_right_side_max_tolerance=self.opt_roving_reference_confirmation_time_right_side_max_tolerance,
                                opt_confirmation_time_left_side_max_tolerance=self.opt_roving_reference_confirmation_time_left_side_max_tolerance,
                            )
                            # If there are no scores for confirmation, don't confirm (duh).
                            if len(roving_rebaseline_confirmation_scores_df) == 0:
                                general_roving_confirmed = False
                                raw_pira_roving_confirmed = False
                            else:
                                roving_rebaseline_confirmation_scores = np.array(
                                    roving_rebaseline_confirmation_scores_df[
                                        self.edss_score_column_name
                                    ]
                                )

                        # Check general baseline
                        if current_edss < general_baselines.iloc[-1]["baseline_score"]:
                            # If there's no confirmation requirement, we use the current score
                            # as our new baseline.
                            confirmed_new_roving = current_edss
                            # Else we check if it is confirmed, and compute the confirmed score.
                            if self.opt_roving_reference_require_confirmation:
                                # We already have the scores; if they are not empty, check.
                                # Otherwise there's no confirmation anyways.
                                if general_roving_confirmed:
                                    # All confirmation scores musst be lower than the current baseline
                                    if (
                                        max(roving_rebaseline_confirmation_scores)
                                        < general_baselines.iloc[-1]["baseline_score"]
                                    ):
                                        confirmed_new_roving = max(
                                            max(roving_rebaseline_confirmation_scores),
                                            current_edss,
                                        )
                                    else:
                                        general_roving_confirmed = False

                            # If confirmed, append a new baseline to our collection.
                            if general_roving_confirmed:
                                annotated_df.at[
                                    i, self.is_general_rebaseline_flag_column_name
                                ] = True
                                annotated_df.at[
                                    i,
                                    self.used_as_general_reference_score_flag_column_name,
                                ] = confirmed_new_roving
                                general_baselines = pd.concat(
                                    [
                                        general_baselines,
                                        pd.DataFrame(
                                            {
                                                "baseline_score": [
                                                    confirmed_new_roving
                                                ],
                                                "baseline_timestamp": [
                                                    current_timestamp
                                                ],
                                            }
                                        ),
                                    ]
                                ).reset_index(drop=True)

                        # Check RAW/PIRA baseline
                        if current_edss < raw_pira_baselines.iloc[-1]["baseline_score"]:
                            # If there's no confirmation requirement, we use the current score
                            # as our new baseline.
                            confirmed_new_roving = current_edss
                            # Else we check if it is confirmed, and compute the confirmed score.
                            if self.opt_roving_reference_require_confirmation:
                                # We already have the scores; if they are not empty, check.
                                # Otherwise there's no confirmation anyways.
                                if raw_pira_roving_confirmed:
                                    # All confirmation scores musst be lower than the current baseline
                                    if (
                                        max(roving_rebaseline_confirmation_scores)
                                        < raw_pira_baselines.iloc[-1]["baseline_score"]
                                    ):
                                        confirmed_new_roving = max(
                                            max(roving_rebaseline_confirmation_scores),
                                            current_edss,
                                        )
                                    else:
                                        raw_pira_roving_confirmed = False

                            # If confirmed, append a new baseline to our collection.
                            if raw_pira_roving_confirmed:
                                annotated_df.at[
                                    i, self.is_raw_pira_rebaseline_flag_column_name
                                ] = True
                                annotated_df.at[
                                    i,
                                    self.used_as_raw_pira_reference_score_flag_column_name,
                                ] = confirmed_new_roving
                                raw_pira_baselines = pd.concat(
                                    [
                                        raw_pira_baselines,
                                        pd.DataFrame(
                                            {
                                                "baseline_score": [
                                                    confirmed_new_roving
                                                ],
                                                "baseline_timestamp": [
                                                    current_timestamp
                                                ],
                                            }
                                        ),
                                    ]
                                ).reset_index(drop=True)

        return annotated_df

    def add_progression_events_to_follow_up(
        self,
        follow_up_dataframe,
        relapse_timestamps,
    ):
        """TBD"""
        # --------------------------------------------------------------------------------
        # CHECK INPUT DATA AND ARGUMENTS
        # --------------------------------------------------------------------------------
        # Check if follow-up is well-ordered with unambiguous timestamps
        assert pd.api.types.is_numeric_dtype(
            follow_up_dataframe[self.time_column_name]
        ), "Timestamps must be numeric, e.g. an integer number of days after baseline."
        # Assert that the input data are well ordered
        assert (
            follow_up_dataframe[self.time_column_name].is_monotonic_increasing
            and follow_up_dataframe[self.time_column_name].is_unique
        ), "Input data are not well ordered or contain ambiguous timestamps."
        # Check and order the relapse timestamp list
        assert all(
            isinstance(relapse_timestamp, int)
            for relapse_timestamp in relapse_timestamps
        ), "Relapse timestamps must be numeric, e.g. an integer number of days after baseline."
        relapse_timestamps = sorted(relapse_timestamps)

        # --------------------------------------------------------------------------------
        # ANNOTATE PROGRESSION EVENTS TO DATAFRAME
        # --------------------------------------------------------------------------------
        # If we chose 're-baselining only', 'all', or 'never', one call to
        # '_annotate_progression_events' is sufficient. For the 'end' version,
        # we have to check each follow-up twice, first with the default 're-
        # baselining only' option, then we have to check the remainder (after
        # the last event) for undefined progression.
        if self.undefined_progression == "end":
            undefined_progression_to_pass_on = "re-baselining only"
        else:
            undefined_progression_to_pass_on = self.undefined_progression

        # First round
        annotated_df = self._annotate_progression_events(
            follow_up_dataframe=follow_up_dataframe,
            relapse_timestamps=relapse_timestamps,
            undefined_progression=undefined_progression_to_pass_on,
        )

        # If we choose the 'end' option, we now have to check the remainder of the
        # follow-up for undefined progression at all assessments (the 'all general
        # only' option, we already know that there are no more RAW and PIRA by
        # starting from the last progression event).
        if self.undefined_progression == "end":
            # Get the follow up from the last RAW/PIRA event on.
            progression_events_df = annotated_df[
                (annotated_df[self.is_progression_flag_column_name])
                & (
                    annotated_df[self.progression_type_column_name].isin(
                        [
                            self.label_raw,
                            self.label_pira,
                            self.label_pira_confirmed_in_raw_window,
                        ]
                    )
                )
            ]
            # If there are no events, we look at the entire follow-up again, but
            # this time only for general progression at assessments that are not
            # at post-relapse re-baselining assessments. We only need the timestamp
            # and the score columns, the rest is preserved in the annotated df after
            # the first round.
            if len(progression_events_df) == 0:
                remainder_follow_up = annotated_df[
                    [self.time_column_name, self.edss_score_column_name]
                ].copy()
                last_progression_event_id = 0
            # If there are events, we start our search after the last one, and we
            # have to use the confirmed event score of said last event as our new
            # baseline. Since this re-start can only happen at RAW/PIRA events,
            # both the new RAW/PIRA as well as the general baseline is the confirmed
            # event score. Undefined events where the confirmed score is lower than
            # the post-relapse PIRA/RAW baseline are thus not a problem here.
            else:
                # Get the timestamp of the last progression event
                # NOTE: If events are merged, only the first event has the
                # is progression flag set to True. Therefore, we have to use
                # the largest progression event ID instead of the largest
                # timestamp, then get the max timestamp within this progression
                # ID. If we are not using the merge events option, there is only
                # one row with this ID and nothing changes.
                last_progression_event_id = progression_events_df[
                    self.progression_event_id_column_name
                ].max()
                # NOTE Since merged events have no 'is progression' flag, we have
                # to use the full annotated dataframe.
                last_progression_event_timestamp_including_merged = annotated_df[
                    annotated_df[self.progression_event_id_column_name]
                    == last_progression_event_id
                ][self.time_column_name].max()
                # We need the last one as baseline for the remainder,
                # thus the '>=' sign.
                remainder_follow_up = annotated_df[
                    annotated_df[self.time_column_name]
                    >= last_progression_event_timestamp_including_merged
                ].copy()
                # Now get the last general baseline value (the confirmed event score)
                # and use it to overwrite the EDSS of the first assessment so that we
                # start at the correct baseline.
                # NOTE: If we merge events, the reference columns of merged
                # events will be NaN, so we have to get them from the progression
                # event dataframe...
                remainder_general_baseline = progression_events_df[
                    progression_events_df[self.progression_event_id_column_name]
                    == last_progression_event_id
                ][self.used_as_general_reference_score_flag_column_name].max()
                # Also get the RAW/PIRA baseline; they MUST be equal, so we add an
                # assertion that breaks everything if this assumption is wrong or if
                # there is an implementation error...
                # NOTE: we need the .max() since the baseline is not provided
                # for merged events, so iloc[0] might return a NaN.
                remainder_raw_pira_baseline = progression_events_df[
                    progression_events_df[self.progression_event_id_column_name]
                    == last_progression_event_id
                ][self.used_as_raw_pira_reference_score_flag_column_name].max()
                assert (
                    remainder_general_baseline == remainder_raw_pira_baseline
                ), "Check the 'end' option for undefined, inconsistent baselines!"
                # Set the first event of the remainder to the confirmed event score;
                # this will be our new study baseline.
                remainder_follow_up.at[
                    remainder_follow_up.index[0], self.edss_score_column_name
                ] = remainder_general_baseline
                # We only need the timestamp and the score columns, the rest is
                # preserved in the annotated df after the first round.
                remainder_follow_up = remainder_follow_up[
                    [self.time_column_name, self.edss_score_column_name]
                ]
            # Now if the event was at the end of the follow-up, we're done.
            # Otherwise, we now check the remainder follow-up again, and this
            # time we set the 'undefined_progression' arg to the custom arg
            # "all general only", which skips looking for RAW or PIRA. All
            # other args remain the same.
            if len(remainder_follow_up) > 1:
                annotated_df_second_run = self._annotate_progression_events(
                    follow_up_dataframe=remainder_follow_up,
                    relapse_timestamps=relapse_timestamps,
                    undefined_progression="all general only",
                )
                # First, check whether we've found any progression, otherwise we
                # can call it a day.
                progression_events_df_second_run = annotated_df_second_run[
                    annotated_df_second_run[self.is_progression_flag_column_name]
                ].copy()
                if len(progression_events_df_second_run) > 0:
                    # Now we have to overwrite all progression related stuff...
                    # NOTE: we must not overwrite the first row, as this is
                    # our progression event from the first run!
                    for i, row in annotated_df_second_run.iloc[1:].iterrows():
                        annotated_df.at[i, self.is_progression_flag_column_name] = row[
                            self.is_progression_flag_column_name
                        ]
                        annotated_df.at[i, self.progression_type_column_name] = row[
                            self.progression_type_column_name
                        ]
                        annotated_df.at[i, self.progression_score_column_name] = row[
                            self.progression_score_column_name
                        ]
                        annotated_df.at[
                            i, self.progression_reference_score_column_name
                        ] = row[self.progression_reference_score_column_name]
                        annotated_df.at[i, self.progression_event_id_column_name] = (
                            row[self.progression_event_id_column_name]
                            + last_progression_event_id
                        )
                        annotated_df.at[
                            i, self.is_post_event_rebaseline_flag_column_name
                        ] = row[self.is_post_event_rebaseline_flag_column_name]
                        # General
                        annotated_df.at[
                            i, self.is_general_rebaseline_flag_column_name
                        ] = row[self.is_general_rebaseline_flag_column_name]
                        annotated_df.at[
                            i, self.used_as_general_reference_score_flag_column_name
                        ] = row[self.used_as_general_reference_score_flag_column_name]
                        # RAW/PIRA
                        annotated_df.at[
                            i, self.is_raw_pira_rebaseline_flag_column_name
                        ] = row[self.is_raw_pira_rebaseline_flag_column_name]
                        annotated_df.at[
                            i, self.used_as_raw_pira_reference_score_flag_column_name
                        ] = row[self.used_as_raw_pira_reference_score_flag_column_name]

        return annotated_df


if __name__ == "__main__":
    pass
