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
    is_general_rebaseline_flag_column_name: str = "is_general_rebaseline"
    is_post_event_rebaseline_flag_column_name: str = "is_post_event_rebaseline"
    used_as_general_reference_score_column_name: str = (
        "edss_score_used_as_new_general_reference"
    )
    is_event_flag_column_name: str = "is_event"
    is_accrual_flag_column_name: str = "is_accrual_event"
    is_improvement_flag_column_name: str = "is_improvement_event"
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
        if (self.annotation_mode == "experimental-symmetric") and (
            self.opt_baseline_type == "roving"
        ):
            raise ValueError(
                "Roving reference is not available for 'experimental-symmetric' annotation mode."
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
                # TODO: additional threshold
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
                # to or smaller than the candidate.
                # TODO: additional threshold
                elif current_edss < current_reference:
                    if max(confirmation_scores) <= current_edss:
                        is_confirmed_decrease = True
                        confirmed_edss = current_edss

        return is_confirmed_increase, is_confirmed_decrease, confirmed_edss

    def _backtrack_minimal_distance_compatible_reference(
        self,
        current_edss,
        current_timestamp,
        check_increase,
        check_decrease,
        baselines_df,
    ):
        """
        TODO: cover increase and decrease

        """
        assert check_decrease != check_increase, (
            "Can't check both increase and decrease!"
        )
        # From all the previous references, flag those that are
        # low enough so that 'current_edss' would be an accrual
        # with respect to them, or high enough that 'current_edss'
        # would be an improvement.
        previous_rebaselines = baselines_df.copy()
        previous_rebaselines[
            [
                "low_enough_to_be_accrual_reference",
                "high_enough_to_be_improvement_reference",
            ]
        ] = previous_rebaselines.apply(
            lambda row: self._is_large_enough_increase_or_decrease(
                current_edss=current_edss,
                reference_edss=row[self.baseline_score_column_name],
            ),
            result_type="expand",
            axis=1,
        )
        # Get suitable previous references
        if check_increase:
            previous_rebaselines = previous_rebaselines[
                previous_rebaselines["low_enough_to_be_accrual_reference"]
            ].copy()
        elif check_decrease:
            previous_rebaselines = previous_rebaselines[
                previous_rebaselines["high_enough_to_be_improvement_reference"]
            ].copy()
        # If none of the previous references is low or high
        # enough, we're done...
        if len(previous_rebaselines) == 0:
            return np.nan, np.nan
        # ... else we have to get those that also fulfill the
        # minimal distance requirement.
        else:
            suitable_and_far_enough = previous_rebaselines[
                previous_rebaselines[self.baseline_timestamp_column_name]
                + self.opt_minimal_distance_time
                <= current_timestamp
            ].copy()
            if len(suitable_and_far_enough) > 0:
                return (
                    suitable_and_far_enough.iloc[-1][self.baseline_score_column_name],
                    suitable_and_far_enough.iloc[-1][
                        self.baseline_timestamp_column_name
                    ],
                )
            else:
                return np.nan, np.nan

    def _check_assessment_for_progression(
        self,
        annotated_df,
        baselines_df,
        current_assessment_index,
        additional_lower_threshold,
    ):
        """Check if a score is a progression event.

        This function checks if an EDSS score is an event by
        checking the minimal distance, minimal increase or
        decrease conditions, and confirmation conditions.

        Returns acrual/improvement yes/no, type, event score,
        and the reference score for the event.

        TODO: Check for annotation mode!
        annotation_mode: str = (
            "accrual"  # or "experimental-inverted", "experimental-symmetric"
        )

        Args:
        - annotated_df: follow-up dataframe with time from last and
                        time to next relapse
        - baselines_df: dataframe with baselines
        - current_assessment_index: the index of the current assessment
        - additional_lower_threshold: additional threshold for progression

        Returns:
        - bool, bool, bool,
          str, float, float: is_event, is_accrual, is_improvement, event_type,
                                   confirmed_event_score, current_baseline_score
        """
        # Get some scores/timestamps
        row = annotated_df.loc[current_assessment_index]
        current_edss = row[self.edss_score_column_name]
        current_timestamp = row[self.time_column_name]
        current_baseline_score = baselines_df.iloc[-1]["baseline_score"]

        # Set return variables
        is_event = False
        is_accrual = False
        is_improvement = False
        event_type = None
        confirmed_event_score = np.nan

        # Are we looking at improvement or accrual candidate?
        # Sepending on annotation mode, we check increase only
        # or decrease only.
        check_increase = False
        check_decrease = False
        if (current_edss > current_baseline_score) and (
            self.annotation_mode in ["accrual", "experimental-symmetric"]
        ):
            check_increase = True
        elif (current_edss < current_baseline_score) and (
            self.annotation_mode in ["experimental-inverted", "experimental-symmetric"]
        ):
            check_decrease = True

        # If the score is below our additional lower threshold, it is not an
        # event candidate anyways. # TODO: additional upper threshold
        if (
            check_increase and (current_edss >= additional_lower_threshold)
        ) or check_decrease:
            # The minimal distance has to be checked first, since it
            # can change the reference score if we allow backtracking.
            minimal_distance_condition_satisfied = True
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
                            check_increase=check_increase,
                            check_decrease=check_decrease,
                            baselines_df=baselines_df,
                        )
                        if backtracked_timestamp >= 0:
                            distance = current_timestamp - backtracked_timestamp
                            current_baseline_score = backtracked_reference

                if distance < self.opt_minimal_distance_time:
                    minimal_distance_condition_satisfied = False

            # Now that the distance is checked, check if the increase is large enough.
            if minimal_distance_condition_satisfied:
                # Does it qualify as accrual or improvement?
                is_increase, is_decrease = self._is_large_enough_increase_or_decrease(
                    current_edss=current_edss,
                    reference_edss=current_baseline_score,
                )
                if is_increase or is_decrease:
                    # If we don't require confirmation, we're done.
                    if not self.opt_require_confirmation:
                        # Determine event type
                        is_event = True
                        is_accrual = is_increase
                        is_improvement = is_decrease
                        confirmed_event_score = current_edss
                        if is_increase:
                            event_type = self.label_pira
                        elif is_decrease:
                            event_type = self.label_improvement

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
                        is_event = True
                        is_accrual = is_increase
                        is_improvement = is_decrease
                        confirmed_event_score = current_edss
                        if is_increase:
                            event_type = self.label_pira
                        elif is_decrease:
                            event_type = self.label_improvement
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
                            # is_confirmed_increase, is_confirmed_decrease, confirmed_edss
                            (
                                is_accrual,
                                is_improvement,
                                confirmed_event_score,
                            ) = self._check_confirmation_scores_and_get_confirmed_score(
                                current_edss=current_edss,
                                current_reference=current_baseline_score,  # The UP vs. RAW/PIRA version choice happens at the start.
                                confirmation_scores_dataframe=confirmation_scores_dataframe,
                                additional_lower_threshold=additional_lower_threshold,
                            )
                            # If unconfirmed, nope, otherwise continue and check event type
                            # TODO: for accrual, check RAW/PIRA/Undefined
                            if is_accrual or is_improvement:
                                is_event = True
                                if is_increase:
                                    event_type = self.label_pira
                                elif is_decrease:
                                    event_type = self.label_improvement

        return (
            is_event,
            is_accrual,
            is_improvement,
            event_type,
            confirmed_event_score,
            current_baseline_score,
        )

    def _combine_events_forward_lookup(
        self,
        annotated_df,
        baselines_df,
        iid_index,
        iid_confirmed_event_score,
        iid_event_type,
        iid_is_accrual,
        iid_is_improvement,
        additional_lower_threshold,
    ):
        """Find the indices of merged events, the event score, and
        the timestamp of the last event within series of merged events.

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

        This is to identify connected events; we only look at strictly
        monotonically increasing or decreasing scores, with an optional
        tolerance for identical scores recorded in close temporal proximity.

        Notes:
        *   Just a little fluke improvement or accrual already stops this
            process... Show this quirk in the documentation!
        *   Assessments considered as repetition measurements (i.e.
            within continuous_events_max_repetition_time) are also flagged
            as members of the merged event, but not if they are at the end.
        *   This is meant to be used for PIRA/RAW; undefined events are
            always considered singular.
        *   Events included into a merged event series don't get their own
            'is event' flag or a progression type/score/reference. This is
            by design in order to make analysis easier (e.g. event counts
            based on rows with 'is_progression == True'). They can be
            identified via the event ID.

        Args:
        - annotated_df: follow-up dataframe with time from last and
                        time to next relapse
        - baselines_df: dataframe with RAW/PIRA and general baselines
        - relapse_timestamps: list of relapse timestamps
        - iid_index: the index of the first progression event
        - iid_confirmed_event_score: the confirmed score of the first event
        - iid_progression_type: the type of the first event
        - additional_lower_threshold: additional threshold for progression

        Returns:
        - list, float, int: indices_of_merged_event, confirmed_event_score,
                            last_confirmed_progression_timestamp

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
            # If the score is lower (when merging accrual events) or
            # higher (when merging improvement events) than the current
            # confirmed event score, we stop. In this case, any confirmed
            # score would be lower/higher than the previous one anyways.
            if (
                iid_is_accrual
                and (row[self.edss_score_column_name] < confirmed_event_score)
            ) or (
                iid_is_improvement
                and (row[self.edss_score_column_name] > confirmed_event_score)
            ):
                break

            # Else we need to test whether the next score from the
            # next assessment would be an event itself. We use the
            # same baseline as we used for the IID.
            else:
                (
                    new_is_event,
                    new_is_accrual,
                    new_is_improvement,
                    new_event_type,
                    new_confirmed_event_score,
                    _,
                ) = self._check_assessment_for_progression(
                    annotated_df=annotated_df,
                    baselines_df=baselines_df,
                    current_assessment_index=i,
                    additional_lower_threshold=additional_lower_threshold,
                )

                # If the new score is not a progression w.r.t. the IID
                # baseline anymore, we stop the merge. This could happen
                # if e.g. a 'next confirmed' requirement is in place.
                if not new_is_event:
                    break
                # We also have to check whether the progression is
                # still of the same type; otherwise we also stop.
                if new_event_type != iid_event_type:
                    break
                # If the confirmed score is lower than the previous
                # one when merging accrual events, or higher than
                # the previous one when merging improvement events,
                # we consider the merged event over.
                if (
                    iid_is_accrual
                    and (new_confirmed_event_score < confirmed_event_score)
                ) or (
                    iid_is_improvement
                    and (new_confirmed_event_score > confirmed_event_score)
                ):
                    break
                else:
                    # If the new score leads to an increased event score
                    # when merging accural events or a decreased event
                    # score when merging improvement events, we reset the
                    # stagnation flag and clear the IDs of stagnation events,
                    # since they are now not at the end of the merge anymore.
                    if (
                        iid_is_accrual
                        and (new_confirmed_event_score > confirmed_event_score)
                    ) or (
                        iid_is_improvement
                        and (new_confirmed_event_score < confirmed_event_score)
                    ):
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
                additional_lower_threshold_for_progression_and_confirmation = 0

                # Step 1 - is it an event?
                # NOTE: _check_assessment_for_progression also checks for
                # the annotation mode.
                (
                    is_event,
                    is_accrual,
                    is_improvement,
                    event_type,
                    confirmed_event_score,
                    current_baseline_score,
                ) = self._check_assessment_for_progression(
                    annotated_df=annotated_df,
                    baselines_df=general_baselines,
                    current_assessment_index=i,
                    additional_lower_threshold=additional_lower_threshold_for_progression_and_confirmation,
                )
                if is_event:
                    event_id = event_id + 1
                if is_accrual:
                    accrual_event_id = accrual_event_id + 1
                if is_improvement:
                    improvement_event_id = improvement_event_id + 1

                # Step 2 - merge if required
                # If we merge continuous RAW/PIRA/Improvement events: more to come?
                if self.merge_continuous_events:
                    # TODO: add RAW/PIRA in RAW once relapse support is implemented
                    if is_event and (
                        event_type
                        in [
                            self.label_pira,
                            self.label_improvement,
                        ]
                    ):
                        (
                            indices_of_merged_event,
                            confirmed_event_score,
                            last_confirmed_timestamp,
                        ) = self._combine_events_forward_lookup(
                            annotated_df=annotated_df,
                            baselines_df=general_baselines,
                            iid_index=i,
                            iid_confirmed_event_score=confirmed_event_score,
                            iid_event_type=event_type,
                            iid_is_accrual=is_accrual,
                            iid_is_improvement=is_improvement,
                            additional_lower_threshold=0,  # Can't fall below IID score anyway
                        )
                    elif is_event and (
                        event_type
                        not in [
                            self.label_pira,
                            self.label_improvement,
                        ]
                    ):
                        indices_of_merged_event = [i]
                        last_confirmed_timestamp = current_timestamp

                # Step 3 - adjust baselines
                # New baseline? It depends on whether we have found a confirmed event
                # and on whether we are using a roving reference.
                # If there's a progression, we discard all our previous references
                # and continue with the confirmed event score. This will e.g. make
                # checking for the minimal distance with backtracking easier.
                if is_event:
                    # Annotate results...
                    annotated_df.at[i, self.is_event_flag_column_name] = True
                    annotated_df.at[i, self.is_accrual_flag_column_name] = is_accrual
                    annotated_df.at[i, self.is_improvement_flag_column_name] = (
                        is_improvement
                    )
                    annotated_df.at[i, self.event_type_column_name] = event_type
                    annotated_df.at[i, self.event_score_column_name] = (
                        confirmed_event_score
                    )
                    annotated_df.at[i, self.event_reference_score_column_name] = (
                        current_baseline_score
                    )
                    annotated_df.at[i, self.event_id_column_name] = event_id
                    if is_accrual:
                        annotated_df.at[i, self.accrual_event_id_column_name] = (
                            accrual_event_id
                        )
                    elif is_improvement:
                        annotated_df.at[i, self.improvement_event_id_column_name] = (
                            improvement_event_id
                        )
                    # If we merge events: label them.
                    if self.merge_continuous_events:
                        indices_to_skip = indices_to_skip + indices_of_merged_event
                        for event_index in indices_of_merged_event:
                            annotated_df.at[event_index, self.event_id_column_name] = (
                                event_id
                            )
                            if is_accrual:
                                annotated_df.at[
                                    event_index, self.accrual_event_id_column_name
                                ] = accrual_event_id
                            elif is_improvement:
                                annotated_df.at[
                                    event_index, self.improvement_event_id_column_name
                                ] = improvement_event_id

                    # If we only want the first event, we can stop here and we do
                    # not have to bother anymore about baselines...
                    if self.return_first_event_only:
                        break

                    # ... and update baselines. We discard all previous baselines.
                    annotated_df.at[
                        i, self.is_post_event_rebaseline_flag_column_name
                    ] = True
                    # Relapse-independent baseline - this one is reset after any
                    # event irrespective of the event type.
                    annotated_df.at[i, self.is_general_rebaseline_flag_column_name] = (
                        True
                    )
                    annotated_df.at[
                        i, self.used_as_general_reference_score_column_name
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

                # If not an event, check for re-baselining due to roving.
                # TODO: post-relapse re-baselining
                # NOTE: 'else' since this block will contain all other
                # possibilities for a re-baseline.
                else:
                    # If we have a roving baseline, the baselines could improve.
                    # TODO: Write a function for this to avoid all the copying...
                    if self.opt_baseline_type == "roving":
                        general_roving_confirmed = False
                        # Flags for annotation mode
                        check_for_new_lower_reference = False
                        check_for_new_higher_reference = False

                        # Do we even have to check roving reference?
                        if (self.annotation_mode == "accrual") and (
                            current_edss < general_baselines.iloc[-1]["baseline_score"]
                        ):
                            check_for_new_lower_reference = True
                        elif (self.annotation_mode == "experimental-inverted") and (
                            current_edss > general_baselines.iloc[-1]["baseline_score"]
                        ):
                            check_for_new_higher_reference = True

                        # If roving reference requires confirmation, we need
                        # the confirmation scores.
                        if self.opt_roving_reference_require_confirmation and (
                            check_for_new_lower_reference
                            or check_for_new_higher_reference
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

                        # Check the general baseline - lower
                        if check_for_new_lower_reference:
                            if self.opt_roving_reference_require_confirmation:
                                if len(roving_rebaseline_confirmation_scores_df) == 0:
                                    # If there are no scores for confirmation, don't confirm (duh).
                                    general_roving_confirmed = False
                                else:
                                    # All confirmation scores musst be lower than the current baseline
                                    roving_rebaseline_confirmation_scores = np.array(
                                        roving_rebaseline_confirmation_scores_df[
                                            self.edss_score_column_name
                                        ]
                                    )
                                    if (
                                        max(roving_rebaseline_confirmation_scores)
                                        < general_baselines.iloc[-1]["baseline_score"]
                                    ):
                                        confirmed_new_roving = max(
                                            max(roving_rebaseline_confirmation_scores),
                                            current_edss,
                                        )
                                        general_roving_confirmed = True
                                    else:
                                        general_roving_confirmed = False
                            else:
                                # We already know that the current score is lower,
                                # and without a confirmation requirement, we can
                                # use it as new roving reference
                                confirmed_new_roving = current_edss
                                general_roving_confirmed = True

                        # Check the general baseline - higher
                        elif check_for_new_higher_reference:
                            if self.opt_roving_reference_require_confirmation:
                                if len(roving_rebaseline_confirmation_scores_df) == 0:
                                    # If there are no scores for confirmation, don't confirm (duh).
                                    general_roving_confirmed = False
                                else:
                                    # All confirmation scores musst be higher than the current baseline
                                    roving_rebaseline_confirmation_scores = np.array(
                                        roving_rebaseline_confirmation_scores_df[
                                            self.edss_score_column_name
                                        ]
                                    )
                                    if (
                                        min(roving_rebaseline_confirmation_scores)
                                        > general_baselines.iloc[-1]["baseline_score"]
                                    ):
                                        confirmed_new_roving = min(
                                            min(roving_rebaseline_confirmation_scores),
                                            current_edss,
                                        )
                                        general_roving_confirmed = True
                                    else:
                                        general_roving_confirmed = False
                            else:
                                # We already know that the current score is higher,
                                # and without a confirmation requirement, we can
                                # use it as new roving reference
                                confirmed_new_roving = current_edss
                                general_roving_confirmed = True

                        # If confirmed, append a new baseline to our collection.
                        if general_roving_confirmed:
                            annotated_df.at[
                                i, self.is_general_rebaseline_flag_column_name
                            ] = True
                            annotated_df.at[
                                i,
                                self.used_as_general_reference_score_column_name,
                            ] = confirmed_new_roving
                            general_baselines = pd.concat(
                                [
                                    general_baselines,
                                    pd.DataFrame(
                                        {
                                            "baseline_score": [confirmed_new_roving],
                                            "baseline_timestamp": [current_timestamp],
                                        }
                                    ),
                                ]
                            ).reset_index(drop=True)

        return annotated_df

    def add_event_annotation_to_follow_up(
        self,
        follow_up_dataframe,
    ):
        """Add EDSS disability worsening event annotation to
        an EDSS follow-up dataframe.

        ...

        """
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

        # --------------------------------------------------------------------------------
        # ANNOTATE PROGRESSION EVENTS TO DATAFRAME
        # --------------------------------------------------------------------------------

        # First round - NOTE: enough for PIRA vs. Improvement
        annotated_df = self._annotate_events(
            follow_up_dataframe=follow_up_dataframe,
        )

        return annotated_df


if __name__ == "__main__":
    pass
