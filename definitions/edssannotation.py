"""This module contains a class with EDSS disability accrual
or improvement annotation functionality.

See the methods.ipynb notebook in the repo's main folder for
an explanation of the available parameters.

See the tutorial.ipynb notebook in the repo's main folder for
usage examples.

Notes for future features:

-   Warn if events are last confirmed with a confirmation time
    greater than the one for the roving reference

-   Warn of or disable last confirmation in merged mode?

-   General: reduce complexity of args and number of arg combos
    in new default symmetric mode. Don't allow weird stuff such
    as last confirmation.

-   TODO: PIRA and RAW only, PIRA only mode

"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EDSSAnnotation:
    """EDSS disability accrual or improvement event detection
    and classification.

    This class provides the functionality to annotate EDSS
    disability accrual or improvement in a follow-up containing
    EDSS scores and a timestamp for each score.

    Definition options are specified when instantiating the
    class, with our recommendations as default options.

    Default options: symmetric mode (annotates both accrual and
    improvement events), 6 months (180 days) all-confirmed (minimum)
    with respect to a 30-days all-confirmed roving reference, no
    minimal distance requirement, no event merging, no left-hand
    tolerance or right-hand constraints on confirmation time for
    reference or event confirmation, minimum increase + 1.5 for
    score = 0, + 1.0 for scores < 5.5, and + 1 for scores >= 5.5,
    RAW window 30 days pre- and 90 days post-relapse, undefined
    worsening possible at all assessments.

    The effects of individual parameter choices and parameter
    combinations are showcased in the methods.ipynb notebook
    in the repo's main folder.

    To annotate disability accrual or improvement events in a
    follow-up, create an instance of EDSSProgression, then use
    the add_progression_events_to_follow_up method that takes a
    pandas dataframe with a follow-up (at least two columns,
    one for the timestamps and one for the EDSS scores) and
    a list of relapse timestamps to get the annotated dataframe.

    See the tutorial.ipynb notebook in the repo's main folder
    for some input data format and usage examples.

    """

    # Options for annotation mode
    annotation_mode: str = "symmetric"  # or "accrual", "improvement"
    # Options for undefined events
    undefined_events_annotation_mode: str = "all"  # or "re-baselining only", "never"
    # Search mode options
    return_first_event_only: bool = False
    merge_continuous_events: bool = False
    continuous_events_max_repetition_time: int = 30
    continuous_events_max_merge_distance: int = (
        np.inf
    )  # Be more conservative for sparse follow-ups!
    # Baseline options
    opt_baseline_type: str = "fixed"  # "roving"
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
    # PIRA/RAW options - ignored if no relapses specified
    opt_raw_before_relapse_max_time: int = 30
    opt_raw_after_relapse_max_time: int = 90
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
    is_pira_rebaseline_flag_column_name: str = "is_pira_rebaseline"
    is_post_relapse_rebaseline_flag_column_name: str = "is_post_relapse_rebaseline"
    is_post_event_rebaseline_flag_column_name: str = "is_post_event_rebaseline"
    used_as_general_reference_score_column_name: str = (
        "edss_score_used_as_new_general_reference"
    )
    used_as_pira_reference_score_column_name: str = (
        "edss_score_used_as_new_pira_reference"
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
    label_pira_confirmed_in_raw_window: str = "PIRA with relapse during confirmation"
    label_raw: str = "RAW"
    label_undefined_progression: str = "Undefined"
    label_improvement: str = "Improvement"  # Only one type for now

    def __post_init__(self):
        """Non-boilerplate __init__ part."""
        # --------------------------------------------------------------------------
        # String arguments to self
        # --------------------------------------------------------------------------
        self.accrual_mode_name = "accrual"
        self.improvement_mode_name = "improvement"
        self.symmetric_mode_name = "symmetric"
        self.undefined_events_mode_all_name = "all"
        self.undefined_events_mode_rebaselining_only_name = "re-baselining only"
        self.undefined_events_mode_never_name = "never"
        self.undefined_events_mode_all_name = "all"
        self.fixed_baseline_name = "fixed"
        self.roving_reference_name = "roving"
        self.confirmation_all_included_name = "all"
        self.confirmation_last_only_name = "last"
        self.confirmation_condition_minimum_name = "minimum"
        self.confirmation_condition_monotonic_name = "monotonic"
        self.minimal_distance_type_reference_name = "reference"
        self.minimal_distance_type_previous_name = "previous"
        # --------------------------------------------------------------------------
        # Check argument values
        # --------------------------------------------------------------------------
        if self.annotation_mode not in [
            self.accrual_mode_name,
            self.improvement_mode_name,
            self.symmetric_mode_name,
        ]:
            raise ValueError(
                "Invalid annotation mode! Available options: '"
                + self.accrual_mode_name
                + "', '"
                + self.improvement_mode_name
                + "', '"
                + self.symmetric_mode_name
                + "'."
            )
        if self.undefined_events_annotation_mode not in [
            self.undefined_events_mode_all_name,
            self.undefined_events_mode_rebaselining_only_name,
            self.undefined_events_mode_never_name,
        ]:
            raise ValueError(
                "Invalid undefined events annotation mode! Available options: '"
                + self.undefined_events_mode_all_name
                + "', '"
                + self.undefined_events_mode_rebaselining_only_name
                + "', '"
                + self.undefined_events_mode_never_name
                + "'."
            )
        if (self.annotation_mode == self.symmetric_mode_name) and (
            self.opt_baseline_type == self.roving_reference_name
        ):
            raise ValueError(
                "Roving reference is not available for '"
                + self.symmetric_mode_name
                + "' annotation mode."
            )
        if self.merge_continuous_events and (
            self.continuous_events_max_repetition_time < 0
        ):
            raise ValueError("Max. repetition time for merging events must be >= 0.")
        # Baseline arguments
        if self.opt_baseline_type not in [
            self.fixed_baseline_name,
            self.roving_reference_name,
        ]:
            raise ValueError(
                "Invalid baseline option! Available options: '"
                + self.fixed_baseline_name
                + "', '"
                + self.roving_reference_name
                + "'."
            )
        if (
            (self.opt_baseline_type == self.roving_reference_name)
            and (self.opt_roving_reference_require_confirmation)
            and (self.opt_roving_reference_confirmation_time <= 0)
        ):
            raise ValueError(
                "Invalid input for confirmation interval. If confirmation of roving reference required, choose a duration > 0."
            )
        if self.opt_roving_reference_confirmation_included_values not in [
            self.confirmation_all_included_name,
            self.confirmation_last_only_name,
        ]:
            raise ValueError(
                "Invalid option for roving reference confirmation included values! Available options: '"
                + self.confirmation_all_included_name
                + "', '"
                + self.confirmation_last_only_name
                + "'."
            )
        if (self.opt_baseline_type == self.roving_reference_name) and (
            self.opt_roving_reference_confirmation_time_right_side_max_tolerance < 0
        ):
            raise ValueError(
                "Roving reference confirmation right tolerance must be >= 0."
            )
        if (self.opt_baseline_type == self.roving_reference_name) and (
            self.opt_roving_reference_confirmation_time_left_side_max_tolerance < 0
        ):
            raise ValueError(
                "Roving reference confirmation left tolerance must be >= 0."
            )
        # RAW/PIRA arguments
        if self.opt_raw_before_relapse_max_time < 0:
            raise ValueError("Max. RAW time before relapse must be >= 0.")
        if self.opt_raw_after_relapse_max_time < 0:
            raise ValueError("Max. RAW time after relapse must be >= 0.")
        # Confirmation arguments
        if self.opt_require_confirmation:
            if (self.opt_confirmation_time != -1) and (self.opt_confirmation_time <= 0):
                raise ValueError(
                    "Invalid input for confirmation interval. If confirmation required, choose -1 for sustained or a duration > 0."
                )
            if (self.opt_confirmation_time == -1) and (
                self.opt_confirmation_included_values
                == self.confirmation_last_only_name
            ):
                raise ValueError(
                    "Invalid confirmation requirements. For sustained progession, only the option 'all' is valid for included values."
                )
        if self.opt_confirmation_type not in [
            self.confirmation_condition_minimum_name,
            self.confirmation_condition_monotonic_name,
        ]:
            raise ValueError(
                "Invalid confirmation criterion type. Options are '"
                + self.confirmation_condition_minimum_name
                + "' or '"
                + self.confirmation_condition_monotonic_name
                + "'."
            )
        if self.opt_confirmation_included_values not in [
            self.confirmation_all_included_name,
            self.confirmation_last_only_name,
        ]:
            raise ValueError(
                "Invalid confirmation included scores type. Options are "
                + self.confirmation_all_included_name
                + "' or '"
                + self.confirmation_last_only_name
                + "'."
            )
        if self.opt_confirmation_sustained_minimal_distance < 0:
            raise ValueError("Minimal distance for sustained must be >= 0.")
        if self.opt_confirmation_time_right_side_max_tolerance < 0:
            raise ValueError("Confirmation right hand side tolerance must be >= 0.")
        if self.opt_confirmation_time_left_side_max_tolerance < 0:
            raise ValueError("Confirmation left hand side tolerance must be >= 0.")
        # Confirmation requirement combos
        if (
            (self.opt_baseline_type == self.roving_reference_name)
            and (self.opt_roving_reference_require_confirmation)
            and (not self.opt_require_confirmation)
        ):
            raise ValueError(
                "Invalid confirmation options. If confirmation for roving reference is required, it must also be required for events."
            )
        if (
            (self.opt_baseline_type == self.roving_reference_name)
            and (self.opt_roving_reference_require_confirmation)
            and (self.opt_require_confirmation)
            and (
                (self.opt_confirmation_time != -1)
                and (
                    self.opt_roving_reference_confirmation_time
                    > self.opt_confirmation_time
                )
            )
        ):
            raise ValueError(
                "Invalid confirmation options. Confirmation time for events must be equal or larger than confirmation time for roving reference."
            )
        # Minimal distance arguments
        if self.opt_minimal_distance_type not in [
            self.minimal_distance_type_reference_name,
            self.minimal_distance_type_previous_name,
        ]:
            raise ValueError(
                "Invalid minimal distance type. Options are '"
                + self.minimal_distance_type_reference_name
                + "' or '"
                + self.minimal_distance_type_reference_name
                + "'."
            )
        if self.opt_minimal_distance_time < 0:
            raise ValueError("Invalid minimal distance time, must be >= 0.")

        # --------------------------------------------------------------------------
        # Set additional variables
        # --------------------------------------------------------------------------
        self.baseline_score_column_name = "baseline_score"
        self.baseline_timestamp_column_name = "baseline_timestamp"
        self.baselines_df_name = "baselines_df"
        self.check_pira_flag_name = "check_pira"
        self.check_raw_flag_name = "check_raw"
        self.check_undefined_flag_name = "check_undefined"

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

        The improvement and symmetric mode use these arguments
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
            follow-up required for 'sustained'. The only requirement
            is that at least one assessment after the event candidate
            exists. A minimal distance can be set via the sustained
            minimal distance argument.

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
        # Get all assessment after the candidate. The candidate itself
        # must not be included, thus '>' in the selection condition.
        assessments_after_event_candidate = follow_up_dataframe[
            follow_up_dataframe[self.time_column_name] > current_timestamp
        ]
        # If sustained, just take all assessments after the event candidate.
        # If a minimal distance is specified, check whether the resulting
        # remaining dataframe covers the required distance.
        if opt_confirmation_time == -1:
            confirmation_scores_dataframe = assessments_after_event_candidate
            # Check minimal distance; if it is too short, return an empty dataframe.
            if (
                len(
                    assessments_after_event_candidate[
                        assessments_after_event_candidate[self.time_column_name]
                        >= current_timestamp
                        + opt_confirmation_sustained_minimal_distance
                    ]
                )
                == 0
            ):
                confirmation_scores_dataframe = confirmation_scores_dataframe.iloc[0:0]
        # If not, start slicing... Idea: take all assessments >= x after,
        # then obtain the index of the first entry, then for confirmation
        # take all rows from current up to and including this index.
        # NOTE: For next confirmation, choose a tiny interval such as 0.5,
        # don't allow tolerance to the left, and leave the right side
        # unbounded.
        else:
            # Check if the constraint for the maximal distance between
            # an event candidate and the confirmation assessment is met.
            # NOTE: This should technically be called 'at or after event',
            # since at least one event has to be >= confirmation time from
            # candidate to allow confirmatio. This is reflected by the '>='.
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
                if opt_confirmation_included_values == self.confirmation_last_only_name:
                    confirmation_scores_dataframe = confirmation_scores_dataframe.iloc[
                        [-1]
                    ]

        return confirmation_scores_dataframe

    def _check_confirmation_scores_and_get_confirmed_score(
        self,
        current_edss,
        current_reference,
        confirmation_scores_dataframe,
    ):
        """Determines whether an event is confirmed and the
        confirmed event score.

        Looks at confirmatiom scores and checks if they satisfy
        the confirmation conditions (minimal required increase or
        decrease, minimum or monotonic) with respect to the specified
        reference score.

        The confirmation type is loaded from self, since this
        function is only used for confirming events, not for
        confirming baselines.

        The function also returns flags for whether it is a confirmed
        increase or a confirmed decrease.

        Args:
        - current_edss: the current EDSS score
        - current_reference: the current reference score
        - confirmation_scores_dataframe: the confirmation scores

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
            if self.opt_confirmation_type == self.confirmation_condition_minimum_name:
                # Increase: the lowest confirmation score must satisfy
                # the minimal increase condition.
                if (current_edss > current_reference) and (
                    self._is_large_enough_increase_or_decrease(
                        current_edss=min(confirmation_scores),
                        reference_edss=current_reference,
                    )[0]
                ):
                    is_confirmed_increase = True
                    confirmed_edss = min(current_edss, min(confirmation_scores))
                # Decrease: the highest confirmation score must satisfy
                # the minimal decrease condition.
                elif (current_edss < current_reference) and (
                    self._is_large_enough_increase_or_decrease(
                        current_edss=max(confirmation_scores),
                        reference_edss=current_reference,
                    )[1]
                ):
                    is_confirmed_decrease = True
                    confirmed_edss = max(current_edss, max(confirmation_scores))
            elif (
                self.opt_confirmation_type == self.confirmation_condition_monotonic_name
            ):
                # Increase: the lowest confirmation score must be equal
                # to or larger than the candidate.
                if (current_edss > current_reference) and (
                    min(confirmation_scores) >= current_edss
                ):
                    is_confirmed_increase = True
                    confirmed_edss = current_edss
                # Decrease: the highest confirmation score must be equal
                # to or smaller than the candidate.
                elif (current_edss < current_reference) and (
                    max(confirmation_scores) <= current_edss
                ):
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
        """Searches previous reference for references that are low
        enough for the current EDSS to be an accrual candidate with
        respect to them, or high enough for the current EDSS to be an
        improvement candidate with respect to them, and that satisfy
        the minimal distance condition.

        Returns the backtracked reference's score and timestamp if
        one exists, otherwise returns nan.

        This function is only required when using a roving reference,
        not for a fixed baseline. It is thus not used for the symmetric
        annotation mode.

        The idea of this function is that in some cases an increase
        is preceded by a decrease which leads to a re-baselining too
        close to the potential increase. In this case, it would not
        be an accrual, however it would have been if the baseline had
        been stable at a higher level. This might not really make sense
        from a clinical point of view. One solution would of course be
        to require confirmation of a new roving baseline over a time >=
        the minimal distance, but there are published works where the roving
        baseline is NOT confirmed, and even this would not solve the issue
        for all confirmation options.

        References before previous events are NOT allowed as we reset
        the baseline after an event. The same  holds for post-relapse
        re-baselining assessments. However, this can be ensured by
        selecting the 'baselines_df' input appropriately, thus we don't
        have to implement it here.

        The reference only ever decreases for 'accrual' mode and only
        ever increases in 'improvement' mode (overall in relapse-free,
        or per post-relapse or post-event period), so it makes sense to
        take the closest reference to the candidate, because the reference
        is the lowest possible or highest possible, respectively, this way.
        This is important for the confirmation step, where scores need to
        be larger or smaller, respectively, than the reference and some
        increment...

        Args:
        - current_edss: the current EDSS
        - current_timestamp: the current timestamp
        - check_increase: a flag to indicate that the candidate
                          is an increase
        - check_decrease: a flag to indicate that the candidate
                          is a decrease
        - baselines_df: dataframe with all eligible previous baselines

        Returns:
        - float, float: the backtracked reference's score, the
                        backtracked reference's timestamp

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

    def _add_relapses_to_follow_up(
        self,
        follow_up_df,
        relapse_timestamps,
    ):
        """Adds columns with time to next and time since last
        relapse to a follow-up dataframe.

        The relapses are provided via a list of timestamps. From
        this list, we compute for each EDSS assessment the time
        since the last relapse and the time to the next relapse.

        Args:
        - follow_up_df: the follow-up dataframe
        - relapse_timestamps: a list with relapse timestamps

        Returns:
        - dataframe: follow-up dataframe with additional columns
                     for time from last and time to next relapse

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
        """Creates a list with all post-relapse re-baselining
        assessment timestamps.

        We need a function that returns the re-baselining assessment
        timestamps for each relapse. We need to jump through a couple
        of hoops here to correctly account for overlapping RAW windows
        and sequences of relapses with no assessments in between.

        Args:
        - follow_up_df: the follow-up dataframe
        - relapse_timestamps: a list with relapse timestamps

        Returns:
        - list: a list of post-relapse re-baselining timestamps

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
                - self.opt_raw_before_relapse_max_time
            )
            relapses["stop_buffer"] = (
                relapses[relapse_timestamp_column_name]
                + self.opt_raw_after_relapse_max_time
            )
            # Add the start time of the next relapse to each line
            relapses["start_next"] = relapses["start_buffer"].shift(-1)
            # Now find the rebaseline assessment for each relapse
            for _, row in relapses.iterrows():
                rebaseline_candidates = follow_up_df[
                    follow_up_df[self.time_column_name]
                    > row[relapse_timestamp_column_name]
                    + self.opt_raw_after_relapse_max_time
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

    def _check_assessment_for_event(
        self,
        annotated_df,
        relapse_timestamps,
        baselines_df,
        current_assessment_index,
        check_pira,
        check_raw,
        check_undefined,
    ):
        """Check if a score is an event.

        This function checks if an EDSS score is an event by
        checking the minimal distance, minimal increase or
        decrease conditions, and confirmation conditions.

        Returns acrual/improvement yes/no, type, event score,
        and the reference score for the event.

        TODO: PIRA/RAW, and PIRA only mode

        Args:
        - annotated_df: follow-up dataframe with time from last and
                        time to next relapse
        - relapse_timestamps: a list with relapse timestamps
        - baselines_df: dataframe with baselines
        - current_assessment_index: the index of the current assessment
        - check_pira: flag, check PIRA
        - check_raw: flag, check RAW
        - check_undefined: flag, check undefined

        Returns:
        - bool, bool, bool,
          str, float, float: is_event, is_accrual, is_improvement, event_type,
                                   confirmed_event_score, current_baseline_score
        """
        # Get some scores/timestamps
        row = annotated_df.loc[current_assessment_index]
        current_edss = row[self.edss_score_column_name]
        current_timestamp = row[self.time_column_name]
        current_baseline_score = baselines_df.iloc[-1][self.baseline_score_column_name]

        # Set return variables
        is_event = False
        is_accrual = False
        is_improvement = False
        event_type = None
        confirmed_event_score = np.nan

        # Are we looking at improvement or accrual candidate?
        # Depending on annotation mode, we check increase only
        # or decrease only. Also, if the check_pira flag is set,
        # we don't check for improvement, and we also don't check
        # for events if the candidate is within a RAW window.
        check_increase = False
        check_decrease = False
        if (current_edss > current_baseline_score) and (
            self.annotation_mode in [self.accrual_mode_name, self.symmetric_mode_name]
        ):
            # If we are in PIRA mode and the candidate is in a
            # RAW window, we can stop. The check for RAW will be
            # run with the second call to this function.
            if check_pira and (
                (
                    row[self.time_to_next_relapse_column_name]
                    <= self.opt_raw_before_relapse_max_time
                )
                or (
                    row[self.time_since_last_relapse_column_name]
                    <= self.opt_raw_after_relapse_max_time
                )
            ):
                check_increase = False
            else:
                check_increase = True
        elif (
            (not check_pira)  # We don't check for improvement w.r.t. PIRA baseline
            and (current_edss < current_baseline_score)
            and (
                self.annotation_mode
                in [self.improvement_mode_name, self.symmetric_mode_name]
            )
        ):
            check_decrease = True

        # If we don't have to check increase or decrease, we're done.
        if check_increase or check_decrease:
            # The minimal distance has to be checked first, since it
            # can change the reference score if we allow backtracking.
            minimal_distance_condition_satisfied = True
            if self.opt_minimal_distance_time > 0:
                if (
                    self.opt_minimal_distance_type
                    == self.minimal_distance_type_previous_name
                ):
                    previous_timestamp = annotated_df.loc[current_assessment_index - 1][
                        self.time_column_name
                    ]
                    distance = current_timestamp - previous_timestamp
                elif (
                    self.opt_minimal_distance_type
                    == self.minimal_distance_type_reference_name
                ):
                    distance = (
                        current_timestamp
                        - baselines_df.iloc[-1][self.baseline_timestamp_column_name]
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
                    # Determine the event type. NOTE: this might then still
                    # change during confirmation, where PIRA can change to
                    # PIRA with relapse during confirmation.
                    if is_increase:
                        # If we are checking for PIRA, we already know that the
                        # candidate is not in a RAW window, so we can set the
                        # event type to PIRA. It can later be changed to PIRA
                        # with relapse during confirmation when we check the
                        # confirmation condition.
                        if check_pira:
                            event_type = self.label_pira
                        # If we check for RAW, we have to check whether the
                        # candidate is in a RAW window (-> RAW) or not (->
                        # undefined worsening).
                        elif check_raw and (
                            (
                                row[self.time_to_next_relapse_column_name]
                                <= self.opt_raw_before_relapse_max_time
                            )
                            or (
                                row[self.time_since_last_relapse_column_name]
                                <= self.opt_raw_after_relapse_max_time
                            )
                        ):
                            event_type = self.label_raw
                        # If not RAW, it's undefined.
                        else:
                            event_type = self.label_undefined_progression

                    # For decrease, we have only one type.
                    elif is_decrease:
                        event_type = self.label_improvement

                    # Now check confirmation.
                    # If we don't annotate undefined events, we have to exit.
                    if (not check_undefined) and (
                        event_type == self.label_undefined_progression
                    ):
                        event_type = None
                    # If we don't require confirmation, we're done.
                    elif not self.opt_require_confirmation:
                        # Just set the flags and the event score.
                        is_event = True
                        is_accrual = is_increase
                        is_improvement = is_decrease
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
                        is_event = True
                        is_accrual = is_increase
                        is_improvement = is_decrease
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
                            # is_confirmed_increase, is_confirmed_decrease, confirmed_edss
                            (
                                is_accrual,
                                is_improvement,
                                confirmed_event_score,
                            ) = self._check_confirmation_scores_and_get_confirmed_score(
                                current_edss=current_edss,
                                current_reference=current_baseline_score,  # The UP vs. RAW/PIRA version choice happens at the start.
                                confirmation_scores_dataframe=confirmation_scores_dataframe,
                            )
                            # If unconfirmed, nope, otherwise continue and check event type.
                            if is_accrual or is_improvement:
                                is_event = True
                                # If we already know that it's RAW, undefined, or improvement,
                                # we're done. Otherwise we have to check for relapses during
                                # confirmation, unless there are no relapses, of course...
                                # We also introduce a new type 'PIRA with relapse during confirmation'
                                # to mark events that are outside the RAW window, but confirmed
                                # by assessments during relapses. Treated like PIRA for baselines etc.
                                if (
                                    is_accrual
                                    and (event_type == self.label_pira)
                                    and (len(relapse_timestamps) > 0)
                                ):
                                    # If we don't allow relapses between the event and the
                                    # end of the confirmation interval, or if we consider all
                                    # scores for confirmation, then any event with relapses
                                    # in this time interval is 'PIRA with relapse during confirmation'.
                                    # We check the relapse timestamp list for whether there are
                                    # any in between the assessments. We can use >= and <= since
                                    # if the assessment were at the same time as a relapse, it
                                    # would be RAW anyways...
                                    if (
                                        not self.opt_pira_allow_relapses_between_event_and_confirmation
                                    ) or (
                                        self.opt_confirmation_included_values
                                        == self.confirmation_all_included_name
                                    ):
                                        last_confirmation_score_timestamp = (
                                            confirmation_scores_dataframe[
                                                self.time_column_name
                                            ].max()
                                        )
                                        # If any relapse timestamp >= current and <= last conf + buffer,
                                        # the event is 'PIRA with relapse during confirmation'.
                                        if max(
                                            (
                                                np.array(relapse_timestamps)
                                                >= current_timestamp
                                            )
                                            & (
                                                np.array(relapse_timestamps)
                                                <= last_confirmation_score_timestamp
                                                + self.opt_raw_before_relapse_max_time
                                            )
                                        ):
                                            event_type = (
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
                                        <= self.opt_raw_before_relapse_max_time
                                    ):
                                        event_type = (
                                            self.label_pira_confirmed_in_raw_window
                                        )
                                    if (
                                        confirmation_scores_dataframe.iloc[0][
                                            self.time_since_last_relapse_column_name
                                        ]
                                        <= self.opt_raw_after_relapse_max_time
                                    ):
                                        event_type = (
                                            self.label_pira_confirmed_in_raw_window
                                        )

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
        relapse_timestamps,
        baselines_df,
        rebaselines_list,
        iid_index,
        iid_confirmed_event_score,
        iid_event_type,
        iid_is_accrual,
        iid_is_improvement,
        check_pira,
        check_raw,
        check_undefined,
    ):
        """Find the indices of merged events, the event score, and
        the timestamp of the last event within series of merged events.

        This is to identify connected events; we only look at strictly
        monotonically increasing or decreasing scores, with an optional
        tolerance for identical scores recorded in close temporal proximity,
        which are called repetition measurement; the maximal distance a
        repetition assessment can have from the event assessment can be
        specified via the continuous_events_max_repetition_time argument.

        Notes:
        *   Just a little fluke improvement or accrual already stops this
            process... Show this quirk in the documentation!
        *   Assessments considered as repetition measurements (i.e.
            within continuous_events_max_repetition_time) are also flagged
            as members of the merged event, but NOT if they are at the end.
        *   This is meant to be used for PIRA/RAW or improvement; undefined
            events are always considered singular.
        *   Events included into a merged event series don't get their own
            'is event' flag or a progression type/score/reference. This is
            by design in order to make analysis easier (e.g. event counts
            based on rows with 'is_progression == True'). They can be
            identified via the event ID, which all included assessments get.

        Args:
        - annotated_df: follow-up dataframe with time from last and
                        time to next relapse
        - relapse_timestamps: list of relapse timestamps
        - baselines_df: dataframe with RAW/PIRA and general baselines
        - rebaselines_list: list with all post-relapse re-baselining assessments
        - iid_index: the index of the first progression event
        - iid_confirmed_event_score: the confirmed score of the first event
        - iid_event_type: the type of the first event
        - iid_is_accrual: Flag that is true if the IID is accrual
        - iid_is_improvement: Flag that is true if the IID is improvement
        - check_pira: flag, check PIRA
        - check_raw: flag, check RAW
        - check_undefined: flag, check undefined

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

            # If we are merging PIRA and there is a relapse between
            # events, we stop.
            if (iid_event_type == self.label_pira) and (
                len(
                    [
                        relapse_timestamp
                        for relapse_timestamp in relapse_timestamps
                        if (relapse_timestamp >= last_confirmed_progression_timestamp)
                        and (relapse_timestamp <= row[self.time_column_name])
                    ]
                )
                > 0
            ):
                break

            # If we are merging RAW and there is a relapse free period
            # between events, we stop. Previous plus window or next minus
            # window smaller than same for next event.
            if (iid_event_type == self.label_raw) and (
                len(
                    [
                        rebaseline_timestamp
                        for rebaseline_timestamp in rebaselines_list
                        if (
                            rebaseline_timestamp >= last_confirmed_progression_timestamp
                        )
                        and (rebaseline_timestamp <= row[self.time_column_name])
                    ]
                )
                > 0
            ):
                break

            # If the score is lower (when merging accrual events) or
            # higher (when merging improvement events) than the current
            # confirmed event score, we stop. In this case, any confirmed
            # score would be lower/higher than the previous one anyways.
            # Else we need to test whether the next score from the
            # next assessment would be an event itself. We use the
            # same baseline as we used for the IID.
            if (
                iid_is_accrual
                and (row[self.edss_score_column_name] < confirmed_event_score)
            ) or (
                iid_is_improvement
                and (row[self.edss_score_column_name] > confirmed_event_score)
            ):
                break
            else:
                (
                    new_is_event,
                    _,
                    _,
                    new_event_type,
                    new_confirmed_event_score,
                    _,
                ) = self._check_assessment_for_event(
                    annotated_df=annotated_df,
                    relapse_timestamps=relapse_timestamps,
                    baselines_df=baselines_df,
                    current_assessment_index=i,
                    check_pira=check_pira,
                    check_raw=check_raw,
                    check_undefined=check_undefined,
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
        relapse_timestamps,
    ):
        """Annotates EDSS disability accrual or improvement events
        in a follow-up. This function contains most of the actual
        annotation magic. Called by 'add_event_annotation_to_follow_up'.
        This is a separate function because we might re-implement
        special undefined event modes (like the 'end' mode from earlier
        versiona of the algorithm) later on, which require two calls
        of '_annotate_events'.

        Args:
        - follow_up_dataframe: a dataframe with at least the scores
                               and the corresponding timestamps
        - relapse_timestamps: list with relapse timestamps

        Returns:
        - dataframe: the follow-up dataframe with additional columns
                     for event annotation
        """

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
        # The following flags mark all assessments where the general
        # or PIRA baseline is reset. This always coincides with post-
        # event re-baselining for both references, and with post-relapse
        # re-baselining for the PIRA reference. If we use a roving reference,
        # this also flags all assessment where a new roving reference
        # is set.
        annotated_df[self.is_general_rebaseline_flag_column_name] = False
        # If in accrual or symmetric mode, also set a flag for the PIRA
        # baseline. We don't need a PIRA baseline in improvement mode.
        if self.annotation_mode in [self.accrual_mode_name, self.symmetric_mode_name]:
            annotated_df[self.is_pira_rebaseline_flag_column_name] = False
        # Let's also keep track of the scores that are actually
        # carried forward after a re-baselining (in case of event
        # or baseline confirmation constraints, the new baseline is
        # not equivalent to the EDSS score determined at the assessment...)
        annotated_df[self.used_as_general_reference_score_column_name] = np.nan
        if self.annotation_mode in [self.accrual_mode_name, self.symmetric_mode_name]:
            annotated_df[self.used_as_pira_reference_score_column_name] = np.nan
        # Also initialize columns for event annotation. We keep track of the
        # event, event type, event score, event reference score, and event ID.
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
        # Baselines for RAW, Undefined, and Improvement
        general_baselines = pd.DataFrame(
            {
                self.baseline_score_column_name: [study_baseline_score],
                self.baseline_timestamp_column_name: [study_baseline_timestamp],
            }
        )
        # PIRA baselines are ignored in improvement mode, but we still create it
        # for all modes so that we can use pira_baselines later in if/else blocks.
        # if self.annotation_mode in [self.accrual_mode_name, self.symmetric_mode_name]:
        pira_baselines = pd.DataFrame(
            {
                self.baseline_score_column_name: [study_baseline_score],
                self.baseline_timestamp_column_name: [study_baseline_timestamp],
            }
        )

        # Flag all post-relapse re-baselining assessments. We will
        # later use these flags to identify assessments where we have
        # to check for undefined progression, since they don't qualify
        # for RAW or PIRA. We also need this flag in order to identify
        # assessments with residual post-relapse disability that set
        # a new baseline for PIRA.
        # Initialize the column and just set relapse assessments to True
        # via a list of relapse timestamps if there are relapses. This
        # is not required for improvement mode.
        rebaselines_list = []
        if self.annotation_mode in [self.accrual_mode_name, self.symmetric_mode_name]:
            annotated_df[self.is_post_relapse_rebaseline_flag_column_name] = False
            if len(relapse_timestamps) > 0:
                rebaselines_list = self._get_post_relapse_rebaseline_timestamps(
                    follow_up_df=annotated_df,
                    relapse_timestamps=relapse_timestamps,
                )
                annotated_df.loc[
                    annotated_df[self.time_column_name].isin(rebaselines_list),
                    self.is_post_relapse_rebaseline_flag_column_name,
                ] = True

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

                # Now, check if the row is a post-relapse-rebaseline. Depending
                # on the outcome, we check for undefined progression or RAW/PIRA.
                # The idea here is to select the correct set of baselines. For
                # improvement mode, we ignore post-relapse re-baselining.
                if self.annotation_mode in [
                    self.accrual_mode_name,
                    self.symmetric_mode_name,
                ]:
                    is_rebaseline_assessment = row[
                        self.is_post_relapse_rebaseline_flag_column_name
                    ]
                else:
                    is_rebaseline_assessment = False

                # We also check for residual disability here. If the score after the
                # relapse is higher than the PIRA baseline, it is considered residual
                # disability and will lead to an adjustment of the PIRA baseline.
                is_post_relapse_rebaselining_assessment_with_residual_disability = False
                if (
                    is_rebaseline_assessment
                    and self.annotation_mode
                    in [
                        self.accrual_mode_name,
                        self.symmetric_mode_name,
                    ]
                    and (
                        current_edss
                        > pira_baselines.iloc[-1][self.baseline_score_column_name]
                    )
                ):
                    is_post_relapse_rebaselining_assessment_with_residual_disability = (
                        True
                    )

                # Select the baselines to check
                # Depending on whether the assessment is a post-relapse re-baselining
                # assessment or not, and depending on the chosen annotation mode and
                # the mode for undefined progression, we check for different event
                # types.
                if is_rebaseline_assessment:
                    if (
                        self.undefined_events_annotation_mode
                        == self.undefined_events_mode_never_name
                    ):
                        baselines_to_check = []
                    else:
                        baselines_to_check = [
                            {
                                self.baselines_df_name: general_baselines,
                                self.check_pira_flag_name: False,
                                self.check_raw_flag_name: False,
                                self.check_undefined_flag_name: True,
                            }
                        ]
                else:
                    # In improvement only mode, we don't check PIRA. In accrual,
                    # we test all types if there are relapses, and PIRA only if
                    # there are no relapses. In symmetric mode, we test all types.
                    if self.annotation_mode == self.accrual_mode_name:
                        # Always check PIRA first, then RAW/undefined/improvement
                        # if there are relapses.
                        baselines_to_check = [
                            {
                                self.baselines_df_name: pira_baselines,
                                self.check_pira_flag_name: True,
                                self.check_raw_flag_name: False,
                                self.check_undefined_flag_name: False,
                            }
                        ]
                        # We only have to check for the other types if there
                        # are relapses.
                        if len(relapse_timestamps) > 0:
                            if (
                                self.undefined_events_annotation_mode
                                == self.undefined_events_mode_all_name
                            ):
                                baselines_to_check = baselines_to_check + [
                                    {
                                        self.baselines_df_name: general_baselines,
                                        self.check_pira_flag_name: False,
                                        self.check_raw_flag_name: True,
                                        self.check_undefined_flag_name: True,
                                    },
                                ]
                            else:
                                baselines_to_check = baselines_to_check + [
                                    {
                                        self.baselines_df_name: general_baselines,
                                        self.check_pira_flag_name: False,
                                        self.check_raw_flag_name: True,
                                        self.check_undefined_flag_name: False,
                                    },
                                ]
                    elif self.annotation_mode == self.symmetric_mode_name:
                        # Always check PIRA first.
                        baselines_to_check = [
                            {
                                self.baselines_df_name: pira_baselines,
                                self.check_pira_flag_name: True,
                                self.check_raw_flag_name: False,
                                self.check_undefined_flag_name: False,
                            }
                        ]
                        # We also have to check the general baseline in
                        # order to find improvements. But maybe not for
                        # undefined events.
                        if (
                            self.undefined_events_annotation_mode
                            == self.undefined_events_mode_all_name
                        ):
                            baselines_to_check = baselines_to_check + [
                                {
                                    self.baselines_df_name: general_baselines,
                                    self.check_pira_flag_name: False,
                                    self.check_raw_flag_name: True,
                                    self.check_undefined_flag_name: True,
                                },
                            ]
                        else:
                            baselines_to_check = baselines_to_check + [
                                {
                                    self.baselines_df_name: general_baselines,
                                    self.check_pira_flag_name: False,
                                    self.check_raw_flag_name: True,
                                    self.check_undefined_flag_name: False,
                                },
                            ]
                    elif self.annotation_mode == self.improvement_mode_name:
                        # Don't look for PIRA at all.
                        baselines_to_check = [
                            {
                                self.baselines_df_name: general_baselines,
                                self.check_pira_flag_name: False,
                                self.check_raw_flag_name: False,
                                self.check_undefined_flag_name: False,
                            },
                        ]

                # Step 1 - is it an event?
                # NOTE: _check_assessment_for_event also checks for the annotation mode.
                # If we want to check for events, we now run the corresponding function.
                # We loop over all options (to enable the 'all' option for undefined worsening),
                # but break the loop once an event is found. Thus, if we use the 'all' option,
                # we don't overwrite RAW/PIRA events.
                for baseline_option in baselines_to_check:
                    (
                        is_event,
                        is_accrual,
                        is_improvement,
                        event_type,
                        confirmed_event_score,
                        current_baseline_score,
                    ) = self._check_assessment_for_event(
                        annotated_df=annotated_df,
                        relapse_timestamps=relapse_timestamps,
                        baselines_df=baseline_option[self.baselines_df_name],
                        current_assessment_index=i,
                        check_pira=baseline_option[self.check_pira_flag_name],
                        check_raw=baseline_option[self.check_raw_flag_name],
                        check_undefined=baseline_option[self.check_undefined_flag_name],
                    )
                    if is_accrual:
                        accrual_event_id = accrual_event_id + 1
                    if is_improvement:
                        improvement_event_id = improvement_event_id + 1
                    if is_event:
                        event_id = event_id + 1
                        break

                # Step 2 - merge if required
                # If we merge continuous RAW/PIRA/Improvement events: more to come?
                if self.merge_continuous_events:
                    if is_event and (
                        event_type
                        in [
                            self.label_pira,
                            self.label_raw,
                            self.label_pira_confirmed_in_raw_window,
                            self.label_improvement,
                        ]
                    ):
                        check_pira_in_merge = False
                        check_raw_in_merge = False
                        if event_type in [
                            self.label_pira,
                            self.label_pira_confirmed_in_raw_window,
                        ]:
                            check_pira_in_merge = True
                        elif event_type in [self.label_raw, self.label_improvement]:
                            check_raw_in_merge = True
                        (
                            indices_of_merged_event,
                            confirmed_event_score,
                            last_confirmed_timestamp,
                        ) = self._combine_events_forward_lookup(
                            annotated_df=annotated_df,
                            relapse_timestamps=relapse_timestamps,
                            baselines_df=baseline_option[self.baselines_df_name],
                            rebaselines_list=rebaselines_list,
                            iid_index=i,
                            iid_confirmed_event_score=confirmed_event_score,
                            iid_event_type=event_type,
                            iid_is_accrual=is_accrual,
                            iid_is_improvement=is_improvement,
                            check_pira=check_pira_in_merge,
                            check_raw=check_raw_in_merge,
                            check_undefined=False,
                        )
                    elif is_event and (event_type == self.label_undefined_progression):
                        indices_of_merged_event = [i]
                        last_confirmed_timestamp = current_timestamp

                # Step 3 - adjust baselines
                # New baseline? It depends on whether we have found a confirmed event
                # and on whether we are using a roving reference.
                # If there's an event, we discard all our previous references and
                # continue with the confirmed event score. This will e.g. make
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

                        # We have to remove the post-relapse re-baselining flag if there
                        # are assessments initially flagged as post-relapse re-baselining
                        # that are now within a merged improvement event. This problem
                        # does no occur in improvement mode (no post-relapse re-baselining)
                        # or for accrual types (post-relapse re-baselining stops event
                        # merging).
                        if (event_type == self.label_improvement) and (
                            self.annotation_mode == self.symmetric_mode_name
                        ):
                            for event_index in indices_of_merged_event:
                                annotated_df.at[
                                    event_index,
                                    self.is_post_relapse_rebaseline_flag_column_name,
                                ] = False

                    # If we only want the first event, we can stop here and we do
                    # not have to bother anymore about baselines...
                    if self.return_first_event_only:
                        break

                    # ... and update baselines. We discard all previous baselines
                    # so that they are not accessible to the backtracking function.
                    annotated_df.at[
                        i, self.is_post_event_rebaseline_flag_column_name
                    ] = True

                    # Relapse-independent baseline
                    annotated_df.at[i, self.is_general_rebaseline_flag_column_name] = (
                        True
                    )
                    annotated_df.at[
                        i, self.used_as_general_reference_score_column_name
                    ] = confirmed_event_score
                    general_baseline_timestamp = current_timestamp
                    # If we merge events, we keep the timestamp of the last
                    # event within the merge as our baseline timestamp. This
                    # is the timestamp used to check for minimal distance.
                    if self.merge_continuous_events:
                        general_baseline_timestamp = last_confirmed_timestamp
                    general_baselines = pd.DataFrame(
                        {
                            self.baseline_score_column_name: [confirmed_event_score],
                            self.baseline_timestamp_column_name: [
                                general_baseline_timestamp
                            ],
                        }
                    )

                    # RAW/PIRA baseline.
                    if self.annotation_mode in [
                        self.accrual_mode_name,
                        self.symmetric_mode_name,
                    ]:
                        annotated_df.at[i, self.is_pira_rebaseline_flag_column_name] = (
                            True
                        )
                        annotated_df.at[
                            i, self.used_as_pira_reference_score_column_name
                        ] = confirmed_event_score
                        pira_baseline_timestamp = current_timestamp
                        # If we merge events, we keep the timestamp of the last
                        # event within the merge as our baseline timestamp. This
                        # is the timestamp used to check for minimal distance.
                        if self.merge_continuous_events:
                            pira_baseline_timestamp = last_confirmed_timestamp
                        pira_baselines = pd.DataFrame(
                            {
                                self.baseline_score_column_name: [
                                    confirmed_event_score
                                ],
                                self.baseline_timestamp_column_name: [
                                    pira_baseline_timestamp
                                ],
                            }
                        )

                # If not an event, check for re-baselining due to roving.
                # TODO: post-relapse re-baselining
                # NOTE: 'else' since this block will contain all other
                # possibilities for a re-baseline.
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
                        annotated_df.at[i, self.is_pira_rebaseline_flag_column_name] = (
                            True
                        )
                        annotated_df.at[
                            i, self.used_as_pira_reference_score_column_name
                        ] = current_edss
                        pira_baselines = pd.DataFrame(
                            {
                                self.baseline_score_column_name: [current_edss],
                                self.baseline_timestamp_column_name: [
                                    current_timestamp
                                ],
                            }
                        )
                    # Remove the post-relapse re-baselining flag otherwise.
                    elif is_rebaseline_assessment and (
                        current_edss
                        <= pira_baselines.iloc[-1][self.baseline_score_column_name]
                    ):
                        annotated_df.at[
                            i, self.is_post_relapse_rebaseline_flag_column_name
                        ] = False

                    # If we have a roving baseline, the baselines could improve.
                    # TODO: Write a function for this to avoid all the copying...
                    if self.opt_baseline_type == self.roving_reference_name:
                        general_roving_confirmed = False
                        pira_roving_confirmed = False
                        # Flags for annotation mode
                        check_for_new_lower_general_reference = False
                        check_for_new_higher_general_reference = False
                        check_for_new_lower_pira_reference = False
                        check_for_new_higher_pira_reference = False

                        # Do we even have to check roving reference?
                        if (self.annotation_mode == self.accrual_mode_name) and (
                            current_edss
                            < general_baselines.iloc[-1][
                                self.baseline_score_column_name
                            ]
                        ):
                            check_for_new_lower_general_reference = True
                        elif (self.annotation_mode == self.improvement_mode_name) and (
                            current_edss
                            > general_baselines.iloc[-1][
                                self.baseline_score_column_name
                            ]
                        ):
                            check_for_new_higher_general_reference = True

                        if (self.annotation_mode == self.accrual_mode_name) and (
                            current_edss
                            < pira_baselines.iloc[-1][self.baseline_score_column_name]
                        ):
                            check_for_new_lower_pira_reference = True

                        # If roving reference requires confirmation, we need
                        # the confirmation scores.
                        if self.opt_roving_reference_require_confirmation and (
                            check_for_new_lower_general_reference
                            or check_for_new_higher_general_reference
                            or check_for_new_lower_pira_reference
                            or check_for_new_higher_pira_reference
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
                        if check_for_new_lower_general_reference:
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
                                        < general_baselines.iloc[-1][
                                            self.baseline_score_column_name
                                        ]
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
                        elif check_for_new_higher_general_reference:
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
                                        > general_baselines.iloc[-1][
                                            self.baseline_score_column_name
                                        ]
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
                                            self.baseline_score_column_name: [
                                                confirmed_new_roving
                                            ],
                                            self.baseline_timestamp_column_name: [
                                                current_timestamp
                                            ],
                                        }
                                    ),
                                ]
                            ).reset_index(drop=True)

                        # Check the RAW/PIRA baseline - lower
                        if check_for_new_lower_pira_reference:
                            if self.opt_roving_reference_require_confirmation:
                                if len(roving_rebaseline_confirmation_scores_df) == 0:
                                    # If there are no scores for confirmation, don't confirm (duh).
                                    pira_roving_confirmed = False
                                else:
                                    # All confirmation scores musst be lower than the current baseline
                                    roving_rebaseline_confirmation_scores = np.array(
                                        roving_rebaseline_confirmation_scores_df[
                                            self.edss_score_column_name
                                        ]
                                    )
                                    if (
                                        max(roving_rebaseline_confirmation_scores)
                                        < pira_baselines.iloc[-1][
                                            self.baseline_score_column_name
                                        ]
                                    ):
                                        confirmed_new_roving = max(
                                            max(roving_rebaseline_confirmation_scores),
                                            current_edss,
                                        )
                                        pira_roving_confirmed = True
                                    else:
                                        pira_roving_confirmed = False
                            else:
                                # We already know that the current score is lower,
                                # and without a confirmation requirement, we can
                                # use it as new roving reference
                                confirmed_new_roving = current_edss
                                pira_roving_confirmed = True

                        # Check the RAW/PIRA baseline - higher
                        elif check_for_new_higher_pira_reference:
                            if self.opt_roving_reference_require_confirmation:
                                if len(roving_rebaseline_confirmation_scores_df) == 0:
                                    # If there are no scores for confirmation, don't confirm (duh).
                                    pira_roving_confirmed = False
                                else:
                                    # All confirmation scores musst be higher than the current baseline
                                    roving_rebaseline_confirmation_scores = np.array(
                                        roving_rebaseline_confirmation_scores_df[
                                            self.edss_score_column_name
                                        ]
                                    )
                                    if (
                                        min(roving_rebaseline_confirmation_scores)
                                        > pira_baselines.iloc[-1][
                                            self.baseline_score_column_name
                                        ]
                                    ):
                                        confirmed_new_roving = min(
                                            min(roving_rebaseline_confirmation_scores),
                                            current_edss,
                                        )
                                        pira_roving_confirmed = True
                                    else:
                                        pira_roving_confirmed = False
                            else:
                                # We already know that the current score is higher,
                                # and without a confirmation requirement, we can
                                # use it as new roving reference
                                confirmed_new_roving = current_edss
                                pira_roving_confirmed = True

                        # If confirmed, append a new baseline to our collection.
                        if pira_roving_confirmed:
                            annotated_df.at[
                                i, self.is_pira_rebaseline_flag_column_name
                            ] = True
                            annotated_df.at[
                                i,
                                self.used_as_pira_reference_score_column_name,
                            ] = confirmed_new_roving
                            pira_baselines = pd.concat(
                                [
                                    pira_baselines,
                                    pd.DataFrame(
                                        {
                                            self.baseline_score_column_name: [
                                                confirmed_new_roving
                                            ],
                                            self.baseline_timestamp_column_name: [
                                                current_timestamp
                                            ],
                                        }
                                    ),
                                ]
                            ).reset_index(drop=True)

        return annotated_df

    def add_event_annotation_to_follow_up(
        self,
        follow_up_dataframe,
        relapse_timestamps,
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
            relapse_timestamps=relapse_timestamps,
        )

        return annotated_df


if __name__ == "__main__":
    pass
