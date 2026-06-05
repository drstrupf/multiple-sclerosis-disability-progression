"""This module contains a class with EDSS disability accrual
or improvement annotation functionality.

PIRA only!

"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class EDSSAnnotation:
    """EDSS disability accrual or improvement event detection
    and classification.
    """

    # Options for annotation mode
    annotation_mode: str = (
        "accrual"  # or "experimental-inverted", "experimental-symmetric"
    )
    # Search mode options
    return_first_event_only: bool = False
    merge_continuous_events: bool = False
    continuous_events_max_repetition_time: int = 30
    continuous_events_max_merge_distance: int = (
        np.inf
    )  # be more conservative for sparse follow-ups!
    # Baseline options
    opt_baseline_type: str = "roving"  # or "fixed"
    opt_roving_reference_require_confirmation: bool = True
    opt_roving_reference_confirmation_time: float = (
        30  # 0.5 would amount to next confirmed
    )
    opt_roving_reference_confirmation_included_values: str = "all"  # "last" or "all"
    opt_roving_reference_confirmation_time_right_side_max_tolerance: int = (
        np.inf
    )  # np.inf for no constraint
    opt_roving_reference_confirmation_time_left_side_max_tolerance: int = (
        0  # 0 for no tolerance
    )
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
    is_post_event_rebaseline_flag_column_name: str = "is_post_event_rebaseline"
    used_as_general_reference_score_column_name: str = (
        "edss_score_used_as_new_general_reference"
    )
    is_event_flag_column_name: str = "is_event"
    is_accrual_flag_column_name: str = "is_accrual"
    is_improvement_flag_column_name: str = "is_improvement"
    event_type_column_name: str = "event_type"
    event_score_column_name: str = "event_score"
    event_reference_score_column_name: str = "event_reference_score"
    event_id_column_name: str = "event_id"
    accrual_event_id_column_name: str = "accrual_event_id"
    improvement_event_id_column_name: str = "improvement_event_id"
    label_pira: str = "PIRA"  # Only one type for now
    label_improvement: str = "Improvement"  # Only one type for now

    def __post_init__(self):
        """Non-boilerplate __init__ part."""
        # --------------------------------------------------------------------------
        # Check argument values
        # --------------------------------------------------------------------------
        # TODO: disable roving for experimental-symmetric
        if self.annotation_mode not in [
            "accrual",
            "experimental-inverted",
            "experimental-symmetric",
        ]:
            raise ValueError(
                "Invalid annotation mode! Available options: 'accrual', 'experimental-inverted', 'experimental-symmetric'."
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

    def _is_large_enough_increase_or_decrease(
        self,
        current_edss,
        reference_edss,
    ):
        """Determine if a score meets the minimum increase or
        decrease condition.

        Uses the arguments
        - self.opt_max_score_that_requires_plus_1
        - self.opt_larger_increment_from_0

        The inverted and symmetric mode use these arguments
        with a flipped sign, i.e. a decrease is large enough
        if it is at least -1.0 from any reference smaller or
        equal to opt_max_score_that_requires_plus_1 + 0.5,
        and at least -0.5 for references above that. With the
        opt_larger_increment_from_0 set to True, a reference
        score of 1.5 requires a minimal decrease of 1.5 (i.e.
        from 1.5 to 0.0 counts as decrease, from 1.5 to 1.0
        does not).

        Increase
        --------

        Standard definition (Müller 2023):
        - baseline EDSS 0.0: increase of at least 1.5
        - baseline EDSS > 0.0 and <= 5.0: increase of at least 1.0
        - baseline EDSS >= 5.5: increase of at least 0.5

        For a minimal increase of 0.5 irrespective of baseline, choose
        opt_larger_increment_from_0=False and opt_max_score_that_requires_plus_1=-1.
        For a minimal increase of 1.0 irrespective of baseline, choose
        opt_larger_increment_from_0=False and opt_max_score_that_requires_plus_1=10.0.

        Decrease
        --------
        Invert standard definition (Müller 2023):
        - baseline EDSS 1.5: decrease of at least 1.5
        - baseline EDSS > 1.5 and <= 5.5: decrease of at least 1.0
        - baseline EDSS >= 6.0: decrease of at least 0.5

        Examples for standard definition:
        - 5.0 -> 5.5 delta not large enough
        - 5.0 -> 6.0 delta large enough
        - 5.5 -> 6.0 delta large enough
        - 5.5 -> 5.0 delta not large enough
        - 6.0 -> 5.0 delta large enough
        - 6.0 -> 5.5 delta large enough

        Args:
        - current_edss: an EDSS score
        - reference_edss: the reference to which current_edss is compared

        Returns:
        - bool, bool: True if large enough increase, decrease

        """
        # In annotation mode 'accrual', we only have to check
        # for increases, in inverted only for decreases, and
        # in symmetric for both.
        # Prepare return variables
        is_increase = False
        is_decrease = False
        # Check increase if relevant
        if current_edss > reference_edss:
            if self.opt_larger_increment_from_0 and reference_edss == 0:
                minimal_increase = 1.5
            else:
                if reference_edss <= self.opt_max_score_that_requires_plus_1:
                    minimal_increase = 1
                else:
                    minimal_increase = 0.5
            if current_edss >= reference_edss + minimal_increase:
                is_increase = True
        # Check decrease if relevant
        elif current_edss < reference_edss:
            if self.opt_larger_increment_from_0 and reference_edss <= 1.5:
                minimal_decrease = 1.5
            else:
                if reference_edss <= self.opt_max_score_that_requires_plus_1 + 0.5:
                    minimal_decrease = 1
                else:
                    minimal_decrease = 0.5
            if current_edss <= reference_edss - minimal_decrease:
                is_decrease = True
        return is_increase, is_decrease

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
        """This method returns the part of the follow-up dataframe
        that is relevant for confirmation.

        Returns the first score that satisfies the minimal confirmation
        distance condition plus - if opt_confirmation_included_values is
        set to "all" or confirmation is sustained - all scores between
        event candidate and this score.

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
            This can be restricted using the right side max tolerance
            argument (default is infinite) such that events that are
            after confirmation time plus this tolerance will not be
            considered confirmation assessments.
        -   If no right-hand constraint is given, the argument for left
            hand tolerance amounts to setting the confirmation time
            to confirmation time - tolerance. With a right-hand constraint
            using a left-hand tolerance and reducing the confirmation
            time are NOT equivalent.
        -   By default, there is no minimum duration of post-event
            follow-up required for 'sustained'. Such a minimal distance
            can be set via the sustained minimal distance argument.

        Args:
        - current_timestamp: the current score's timestamp
        - follow_up_dataframe: the dataframe with the entire follow-up
        - opt_confirmation_time: the minimal confirmation time
        - opt_confirmation_included_values: included values option
        - opt_confirmation_sustained_minimal_distance: minimal distance for sustained
        - opt_confirmation_time_right_side_max_tolerance: right-hand constraint
        - opt_confirmation_time_left_side_max_tolerance: left-hand tolerance

        Returns:
        - dataframe: part of the original follow-up dataframe that
                     is relevant for confirmation

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
        """Determines whether an event is confirmed and the
        confirmed event score.

        Looks at confirmatiom scores and checks if they satisfy
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

        Args:
        - current_edss: the current EDSS score
        - current_reference: the current reference score
        - confirmation_scores_dataframe: the confirmation scores
        - additional_lower_threshold: additional threshold

        Returns:
        - bool, bool, float: confirmed increase flag,
                             confirmed decrease flag,
                             confirmed score

        """
        is_confirmed_increase = False
        is_confirmed_decrease = False
        confirmed_edss = np.nan
        # Make this function safe for empty confirmation dataframes.
        # We will check this before calling this function, but just in case...
        if len(confirmation_scores_dataframe) > 0:
            confirmation_scores = np.array(
                confirmation_scores_dataframe[self.edss_score_column_name]
            )
            if self.opt_confirmation_type == "minimum":
                # Increase: the lowest confirmation score must satisfy
                # the minimal increase condition, and also meet the
                # optional additional lower threshold condition.
                if current_edss > current_reference:
                    if self._is_large_enough_increase_or_decrease(
                        current_edss=min(confirmation_scores),
                        reference_edss=current_reference,
                    )[0] and (min(confirmation_scores) >= additional_lower_threshold):
                        is_confirmed_increase = True
                        confirmed_edss = min(current_edss, min(confirmation_scores))
                # Decrease: the highest confirmation score must satisfy
                # the minimal decrease condition. Additional threshold
                # not yet implemented.
                elif current_edss < current_reference:
                    if self._is_large_enough_increase_or_decrease(
                        current_edss=max(confirmation_scores),
                        reference_edss=current_reference,
                    )[1]:
                        is_confirmed_decrease = True
                        confirmed_edss = max(current_edss, max(confirmation_scores))
            elif self.opt_confirmation_type == "monotonic":
                # Increase: the lowest confirmation score must be equal
                # to or larger than the candidate, and also meet the
                # optional additional lower threshold condition.
                if current_edss > current_reference:
                    if (min(confirmation_scores) >= current_edss) and (
                        min(confirmation_scores) >= additional_lower_threshold
                    ):
                        is_confirmed_increase = True
                        confirmed_edss = current_edss
                # Decrease: the highest confirmation score must be equal
                # to or smaller than the candidate, and also meet the
                # optional additional lower threshold condition.
                elif current_edss < current_reference:
                    if max(confirmation_scores) <= current_edss:
                        is_confirmed_decrease = True
                        confirmed_edss = current_edss

        return is_confirmed_increase, is_confirmed_decrease, confirmed_edss

    def _annotate_events(
        self,
        follow_up_dataframe,
    ):
        """..."""

        # Prepare the return dataframe
        annotated_df = follow_up_dataframe.copy()

        # Let's keep track of the baselines for easier debugging...
        # The following flag marks post-event re-baselining events.
        annotated_df[self.is_post_event_rebaseline_flag_column_name] = False
        # The following flags mark all assessments where the
        # general baseline is reset. This always coincides with
        # post-event re-baselining. If we use a roving reference,
        # this also flags all assessment where a new roving reference
        # is set.
        annotated_df[self.is_general_rebaseline_flag_column_name] = False
        # Let's also keep track of the scores that are actually
        # carried forward after a re-baselining (in case of event
        # or baseline confirmation constraints, the new baseline is
        # not equivalent to the EDSS score determined at the assessment...)
        annotated_df[self.used_as_general_reference_score_column_name] = np.nan
        # Also initialize columns for progression annotation. We keep
        # track of the event, event type, event score, event reference
        # score, and event ID.
        annotated_df[self.is_event_flag_column_name] = False
        annotated_df[self.is_accrual_flag_column_name] = False
        annotated_df[self.is_improvement_flag_column_name] = False
        annotated_df[self.event_type_column_name] = None
        annotated_df[self.event_score_column_name] = np.nan
        annotated_df[self.event_reference_score_column_name] = np.nan
        annotated_df[self.event_id_column_name] = np.nan
        annotated_df[self.accrual_event_id_column_name] = np.nan
        annotated_df[self.improvement_event_id_column_name] = np.nan
        # Initialize the confirmed event ID. Don't add this to the annotated
        # dataframe, we don't want to give any ID to non-events. The ID is
        # set to 0 here and then incremented by + 1 at each event, such that
        # the first event will start with 1. Also, add separate counts for
        # accrual and improvement events.
        event_id = 0
        accrual_event_id = 0
        improvement_event_id = 0

        # Get the study baseline and timestamp, prepare baseline dataframes.
        # We will then append new baselines if they are updated (roving), or
        # discard previous ones in case of an event.
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

        # If we merge events, we keep track of indices we want to skip. This
        # is because the event merging happens inside the loop that checks
        # each assessment for an event, thus assessments that are already
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

                is_event = False
                is_accrual = False
                is_improvement = False
                event_type = None
                confirmed_event_score = np.nan

                # Step 1 - check the minimal distance requirement
                # TODO, pass for now.
                if self.opt_minimal_distance_time > 0:
                    pass

                # Step 2 - check whether the new score is an accrual
                # or improvement candidate by score delta.
                is_increase, is_decrease = self._is_large_enough_increase_or_decrease(
                    current_edss=current_edss,
                    reference_edss=general_baselines["baseline_score"][-1],
                )

                return is_increase, is_decrease

                # Step 3 - check confirmation

                # Step 4 - adjust baselines


if __name__ == "__main__":
    pass
