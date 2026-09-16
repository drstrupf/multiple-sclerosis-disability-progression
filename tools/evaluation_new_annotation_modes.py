"""
This is a collection of tools for evaluating the output
of the EDSS worsening events annotation algorithm.

TODO: Event merging statistics.

Documentation coming soon; have a look at tutorial.ipynb
in the repo's main folder for usage and output examples.

"""

from dataclasses import dataclass
from itertools import chain, combinations

import lifelines
import numpy as np
import pandas as pd


# Survival for right-censored data, Kaplan-Meier
def get_median_survival_time_kaplan_meier(
    times_to_event_df,
    durations_column_name="duration",
    observed_column_name="observed",
    event_name="event",
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
                "n_" + event_name + "s": n_events_observed,
                event_name + "_rate": event_rate,
                "median_time_to_first_" + event_name: median_time_to_event,
                "median_time_to_first_"
                + event_name
                + "_lower_95CI": median_time_to_event_ci.loc[0.5][
                    "KM_estimate_lower_0.95"
                ],
                "median_time_to_first_"
                + event_name
                + "_upper_95CI": median_time_to_event_ci.loc[0.5][
                    "KM_estimate_upper_0.95"
                ],
            }
        ]
    )


@dataclass
class EDSSAnnotationEvaluation:
    # Input
    edss_score_column_name: str = "edss_score"
    time_column_name: str = "days_after_baseline"
    is_event_flag_column_name: str = "is_event"
    is_accrual_event_flag_column_name: str = "is_accrual_event"
    is_improvement_event_flag_column_name: str = "is_improvement_event"
    event_type_column_name: str = "event_type"
    event_score_column_name: str = "event_score"
    event_reference_score_column_name: str = "event_reference_score"
    event_id_column_name: str = "event_id"
    accrual_event_id_column_name: str = "accrual_event_id"
    improvement_event_id_column_name: str = "improvement_event_id"
    label_pira: str = "PIRA"
    label_pira_confirmed_in_raw_window: str = "PIRA with relapse during confirmation"
    label_raw: str = "RAW"
    label_undefined_worsening: str = "Undefined"
    label_improvement: str = "Improvement"
    # Output
    event_score_delta_column_name: str = "event_score_delta"
    total_score_delta_column_name: str = "total_event_score_delta"
    total_accrual_score_delta_column_name: str = "total_accrual_event_score_delta"
    total_improvement_score_delta_column_name: str = (
        "total_improvement_event_score_delta"
    )
    n_events_column_name: str = "total_events"
    n_accrual_events_column_name: str = "total_accrual_events"
    n_improvement_events_column_name: str = "total_improvement_events"

    first_event_prefix: str = "first_event_"
    first_accrual_event_prefix: str = "first_accrual_event_"
    first_improvement_event_prefix: str = "first_improvement_event_"
    first_timestamp_column_name: str = "first_timestamp"
    last_timestamp_column_name: str = "last_timestamp"
    duration_of_follow_up_column_name: str = "duration_of_follow_up"
    n_follow_ups_column_name: str = "n_follow_ups"
    n_follow_ups_with_events_column_name: str = "n_follow_ups_with_events"
    n_follow_ups_with_accrual_events_column_name: str = (
        "n_follow_ups_with_accrual_events"
    )
    n_follow_ups_with_improvement_events_column_name: str = (
        "n_follow_ups_with_improvement_events"
    )
    contribution_to_total_events_column_name: str = "contribution_to_total_events"
    contribution_to_total_delta_column_name: str = "contribution_to_total_delta"
    contribution_to_total_accrual_events_column_name: str = (
        "contribution_to_total_accrual_events"
    )
    contribution_to_total_accrual_delta_column_name: str = (
        "contribution_to_total_accrual_delta"
    )
    contribution_to_total_improvement_events_column_name: str = (
        "contribution_to_total_improvement_events"
    )
    contribution_to_total_improvement_delta_column_name: str = (
        "contribution_to_total_improvement_delta"
    )
    contribution_of_accrual_to_total_events_column_name: str = (
        "contribution_of_accrual_to_total_events"
    )
    contribution_of_improvement_to_total_events_column_name: str = (
        "contribution_of_improvement_to_total_events"
    )
    n_follow_ups_without_events_column_name: str = "n_follow_ups_without_events"
    n_follow_ups_with_accrual_events_only_column_name: str = (
        "n_follow_ups_with_accrual_events_only"
    )
    n_follow_ups_with_improvement_events_only_column_name: str = (
        "n_follow_ups_with_improvement_events_only"
    )
    n_follow_ups_with_accrual_and_improvement_events_column_name: str = (
        "n_follow_ups_with_accrual_and_improvement_events"
    )
    n_follow_ups_with_equal_n_accrual_improvement_column_name: str = (
        "n_follow_ups_with_equal_n_accrual_improvement"
    )
    n_follow_ups_with_more_accrual_than_improvement_column_name: str = (
        "n_follow_ups_with_more_accrual_than_improvement"
    )
    n_follow_ups_with_less_accrual_than_improvement_column_name: str = (
        "n_follow_ups_with_less_accrual_than_improvement"
    )

    n_follow_ups_delta_zero_column_name: str = "n_follow_ups_delta_zero"
    n_follow_ups_positive_delta_column_name: str = "n_follow_ups_positive_delta"
    n_follow_ups_negative_delta_column_name: str = "n_follow_ups_negative_delta"

    combinations_query_column_name: str = "combination_query"
    combinations_of_follow_ups_with_events_column_name: str = (
        "of_follow_ups_with_events"
    )
    n_merged_assessments_column_name: str = "n_merged_assessments"
    n_merged_accrual_assessments_column_name: str = "n_merged_accrual_assessments"
    n_merged_improvement_assessments_column_name: str = (
        "n_merged_improvement_assessments"
    )
    # Helpers
    dummy_id_column_name: str = "dummy_id"
    """TBD"""

    def __post_init__(self):
        """Non-boilerplate __init__ part."""
        # Add argument checks here.

    def get_merge_base(
        self, annotated_follow_ups, get_stats_by_type=True, groupby_ids=None
    ):
        """TBD"""
        # We need at least one ID to group by; introduce dummy ID
        # if none provided.
        if groupby_ids == None:
            annotated_follow_ups[self.dummy_id_column_name] = 0
            groupby_ids = [self.dummy_id_column_name]
        # Merge base with all follow-up IDs, their first and last timestamp,
        # and the duration of follow-up.
        merge_base = (
            annotated_follow_ups[groupby_ids + [self.time_column_name]]
            .groupby(groupby_ids)[self.time_column_name]
            .agg(first_timestamp="min", last_timestamp="max")
            .rename(
                columns={
                    "first_timestamp": self.first_timestamp_column_name,
                    "last_timestamp": self.last_timestamp_column_name,
                }
            )
            .reset_index()
        )
        merge_base[self.duration_of_follow_up_column_name] = (
            merge_base[self.last_timestamp_column_name]
            - merge_base[self.first_timestamp_column_name]
        )
        # If the stats are by type, we create a row per event type
        # for each follow-up via cross join with all event types.
        if get_stats_by_type:
            merge_base = pd.merge(
                left=merge_base,
                right=pd.DataFrame(
                    {
                        self.event_type_column_name: [
                            self.label_pira,
                            self.label_pira_confirmed_in_raw_window,
                            self.label_raw,
                            self.label_undefined_worsening,
                            self.label_improvement,
                        ]
                    }
                ),
                how="cross",
            )
        return merge_base

    def get_events(self, annotated_follow_ups, get_accrual=True, get_improvement=True):
        """TBD, some thoughts:

        -   We get all events by filtering on the 'is event' flag column.
            Note that for merged events only the first event has this
            flag set to True. The events merged to this first event can
            be identified via the event ID.

        -   We also get the delta EDSS for each event. Note that in case
            of merged events the event score provided for the first event
            of a series of merged events (the one with the 'is event'
            flag set to True) is the score for the entire merged event
            series, i.e. there is no sum over event IDs required.
        """
        if get_accrual and get_improvement:
            event_flag_column = self.is_event_flag_column_name
        elif get_accrual:
            event_flag_column = self.is_accrual_event_flag_column_name
        elif get_improvement:
            event_flag_column = self.is_improvement_event_flag_column_name
        else:
            raise ValueError("Choose at least one event type.")
        events = annotated_follow_ups[annotated_follow_ups[event_flag_column]].copy()
        events[self.event_score_delta_column_name] = (
            events[self.event_score_column_name]
            - events[self.event_reference_score_column_name]
        )
        return events

    def get_follow_up_stats(
        self, annotated_follow_ups, get_stats_by_type=True, id_columns=None
    ):
        """TBD, some thoughts:

        -   We want to know the count of events, the distribution of
            event types, and the time to first event etc. for each
            individual follow-up.
        """
        ###### PRELIMINARIES ######
        # Work on copy
        annotated_follow_ups_copy = annotated_follow_ups.copy()

        # Groupby/agg operations drop groups with zero elements,
        # thus we first create a merge base with all follow ups
        # to keep track of them.
        if id_columns == None:
            annotated_follow_ups_copy[self.dummy_id_column_name] = 0
            id_columns = [self.dummy_id_column_name]

        # Make sure it is well ordered
        annotated_follow_ups_copy = (
            annotated_follow_ups_copy.sort_values(
                by=id_columns + [self.time_column_name]
            )
            .reset_index(drop=True)
            .copy()
        )

        # Get the merge base
        merge_base = self.get_merge_base(
            annotated_follow_ups_copy,
            get_stats_by_type=get_stats_by_type,
            groupby_ids=id_columns,
        )

        # Select columns to group by.
        groupby_columns = id_columns
        if get_stats_by_type:
            groupby_columns = groupby_columns + [self.event_type_column_name]

        # Get event dataframes
        all_events = self.get_events(
            annotated_follow_ups=annotated_follow_ups_copy,
            get_accrual=True,
            get_improvement=True,
        )
        if not get_stats_by_type:
            accrual_events = self.get_events(
                annotated_follow_ups=annotated_follow_ups_copy,
                get_accrual=True,
                get_improvement=False,
            )
            improvement_events = self.get_events(
                annotated_follow_ups=annotated_follow_ups_copy,
                get_accrual=False,
                get_improvement=True,
            )

        ###### COUNT EVENTS AND SUM DELTAS ######
        # Select columns to aggregate for counts and deltas.
        counts_deltas_aggregation_columns = [
            self.event_id_column_name,
            self.event_score_delta_column_name,
        ]

        # Helper function
        def _get_counts_deltas(events_df, count_column_name, delta_column_name):
            return (
                events_df[groupby_columns + counts_deltas_aggregation_columns]
                .groupby(groupby_columns)
                .agg(
                    count=(self.event_id_column_name, "count"),
                    total_delta=(self.event_score_delta_column_name, "sum"),
                )
                .reset_index()
                .rename(
                    columns={
                        "count": count_column_name,
                        "total_delta": delta_column_name,
                    }
                )
            )

        # Overall
        event_counts_deltas_overall = _get_counts_deltas(
            events_df=all_events,
            count_column_name=self.n_events_column_name,
            delta_column_name=self.total_score_delta_column_name,
        )
        if not get_stats_by_type:
            # Accrual
            event_counts_deltas_accrual = _get_counts_deltas(
                events_df=accrual_events,
                count_column_name=self.n_accrual_events_column_name,
                delta_column_name=self.total_accrual_score_delta_column_name,
            )
            # Improvement
            event_counts_deltas_improvement = _get_counts_deltas(
                events_df=improvement_events,
                count_column_name=self.n_improvement_events_column_name,
                delta_column_name=self.total_improvement_score_delta_column_name,
            )

        # Merge dataframes and adjust data types.
        stats_df = pd.merge(
            left=merge_base,
            right=event_counts_deltas_overall,
            on=groupby_columns,
            how="left",
        ).fillna(0)
        if not get_stats_by_type:
            stats_df = pd.merge(
                left=stats_df,
                right=event_counts_deltas_accrual,
                on=groupby_columns,
                how="left",
            ).fillna(0)
            stats_df = pd.merge(
                left=stats_df,
                right=event_counts_deltas_improvement,
                on=groupby_columns,
                how="left",
            ).fillna(0)
        stats_df[self.n_events_column_name] = stats_df[
            self.n_events_column_name
        ].astype(int)
        if not get_stats_by_type:
            stats_df[self.n_accrual_events_column_name] = stats_df[
                self.n_accrual_events_column_name
            ].astype(int)
            stats_df[self.n_improvement_events_column_name] = stats_df[
                self.n_improvement_events_column_name
            ].astype(int)

        ###### FIRST EVENTS ######
        # Select columns to aggregate for time to first event.
        first_events_aggregation_columns = [
            self.time_column_name,
            self.event_reference_score_column_name,
            self.event_score_column_name,
            self.event_score_delta_column_name,
        ]
        # If we compute overall stats, keep track of the event
        # type of the first event (accrual or improvement)
        if not get_stats_by_type:
            first_events_aggregation_columns = first_events_aggregation_columns + [
                self.event_type_column_name
            ]

        # Helper function
        def _get_time_to(events_df, prefix_variable_name):
            return events_df.loc[
                events_df.groupby(groupby_columns)[self.time_column_name].idxmin()
            ][groupby_columns + first_events_aggregation_columns].rename(
                columns={
                    column_name: prefix_variable_name + column_name
                    for column_name in first_events_aggregation_columns
                }
            )

        # Overall
        first_events_overall = _get_time_to(
            events_df=all_events, prefix_variable_name=self.first_event_prefix
        )
        if not get_stats_by_type:
            # Accrual
            first_events_accrual = _get_time_to(
                events_df=accrual_events,
                prefix_variable_name=self.first_accrual_event_prefix,
            )
            # Improvement
            first_events_improvement = _get_time_to(
                events_df=improvement_events,
                prefix_variable_name=self.first_improvement_event_prefix,
            )

        # Merge dataframes - leave NaNs where there are no events!
        # NOTE: The NaNs are important for survival analysis later
        # on - they indicate follow-ups without event during the
        # observation period.
        stats_df = pd.merge(
            left=stats_df, right=first_events_overall, on=groupby_columns, how="left"
        )
        if not get_stats_by_type:
            stats_df = pd.merge(
                left=stats_df,
                right=first_events_accrual,
                on=groupby_columns,
                how="left",
            )
            stats_df = pd.merge(
                left=stats_df,
                right=first_events_improvement,
                on=groupby_columns,
                how="left",
            )

        # Drop the dummy ID
        if self.dummy_id_column_name in stats_df.columns:
            stats_df = stats_df.drop(columns=[self.dummy_id_column_name])

        return stats_df

    def get_cohort_stats(
        self,
        stats_by_follow_up,
        get_stats_by_type=True,
        follow_up_id_column=None,
        groupby_columns=None,
    ):
        """TBD"""
        # Follow-up ID, add dummy if not provided
        if follow_up_id_column is None:
            stats_by_follow_up[self.dummy_id_column_name] = 0
            follow_up_id_column = self.dummy_id_column_name
        # Groupby columns
        if groupby_columns == None:
            stats_by_follow_up[self.dummy_id_column_name + "_groupby"] = 0
            groupby_columns = [self.dummy_id_column_name + "_groupby"]

        # Introduce new groupby list that includes types if applicable,
        # while keeping the original list, too, for convenience. We need
        # the groupby list without types later for computing the norms.
        groupby_columns_with_types = groupby_columns
        if get_stats_by_type:
            groupby_columns_with_types = groupby_columns_with_types + [
                self.event_type_column_name
            ]

        # Count follow-ups and get total number of event and total delta EDSS.
        if get_stats_by_type:
            event_counts_deltas = (
                stats_by_follow_up[
                    groupby_columns_with_types
                    + [
                        follow_up_id_column,
                        self.n_events_column_name,
                        self.total_score_delta_column_name,
                    ]
                ]
                .groupby(groupby_columns_with_types)
                .agg(
                    n_follow_ups=(follow_up_id_column, "count"),
                    total_events=(self.n_events_column_name, "sum"),
                    total_delta=(self.total_score_delta_column_name, "sum"),
                )
                .reset_index()
                .rename(
                    columns={
                        "n_follow_ups": self.n_follow_ups_column_name,
                        "total_events": self.n_events_column_name,
                        "total_delta": self.total_score_delta_column_name,
                    }
                )
            )
        else:
            event_counts_deltas = (
                stats_by_follow_up[
                    groupby_columns
                    + [
                        follow_up_id_column,
                        self.n_events_column_name,
                        self.n_accrual_events_column_name,
                        self.n_improvement_events_column_name,
                        self.total_score_delta_column_name,
                        self.total_accrual_score_delta_column_name,
                        self.total_improvement_score_delta_column_name,
                    ]
                ]
                .groupby(groupby_columns)
                .agg(
                    n_follow_ups=(follow_up_id_column, "count"),
                    total_events=(self.n_events_column_name, "sum"),
                    total_accrual_events=(self.n_accrual_events_column_name, "sum"),
                    total_improvement_events=(
                        self.n_improvement_events_column_name,
                        "sum",
                    ),
                    total_delta=(self.total_score_delta_column_name, "sum"),
                    total_accrual_delta=(
                        self.total_accrual_score_delta_column_name,
                        "sum",
                    ),
                    total_improvement_delta=(
                        self.total_improvement_score_delta_column_name,
                        "sum",
                    ),
                )
                .reset_index()
                .rename(
                    columns={
                        "n_follow_ups": self.n_follow_ups_column_name,
                        "total_events": self.n_events_column_name,
                        "total_accrual_events": self.n_accrual_events_column_name,
                        "total_improvement_events": self.n_improvement_events_column_name,
                        "total_delta": self.total_score_delta_column_name,
                        "total_accrual_delta": self.total_accrual_score_delta_column_name,
                        "total_improvement_delta": self.total_improvement_score_delta_column_name,
                    }
                )
            )
        # Get type distributions (i.e. the proportion of events that
        # are of a given type).
        if get_stats_by_type:
            # Count total events and deltas irrespective of type for the norm.
            norm = (
                stats_by_follow_up[
                    groupby_columns
                    + [
                        self.n_events_column_name,
                        self.total_score_delta_column_name,
                    ]
                ]
                .groupby(groupby_columns)
                .agg(
                    events_norm=(self.n_events_column_name, "sum"),
                    delta_norm=(self.total_score_delta_column_name, "sum"),
                )
                .reset_index()
            )
            # Get norms for accrual and improvement
            accrual_norm = (
                stats_by_follow_up[
                    stats_by_follow_up[self.event_type_column_name]
                    != self.label_improvement
                ][
                    groupby_columns
                    + [self.n_events_column_name, self.total_score_delta_column_name]
                ]
                .groupby(groupby_columns)
                .agg(
                    acc_events_norm=(self.n_events_column_name, "sum"),
                    acc_delta_norm=(self.total_score_delta_column_name, "sum"),
                )
                .reset_index()
            )
            improvement_norm = (
                stats_by_follow_up[
                    stats_by_follow_up[self.event_type_column_name]
                    == self.label_improvement
                ][
                    groupby_columns
                    + [self.n_events_column_name, self.total_score_delta_column_name]
                ]
                .groupby(groupby_columns)
                .agg(
                    imp_events_norm=(self.n_events_column_name, "sum"),
                    imp_delta_norm=(self.total_score_delta_column_name, "sum"),
                )
                .reset_index()
            )
            norm_acc_imp = pd.merge(
                left=accrual_norm,
                right=improvement_norm,
                on=groupby_columns,
                how="inner",
            )
            # Overall
            event_counts_deltas = pd.merge(
                left=event_counts_deltas, right=norm, on=groupby_columns, how="left"
            )
            event_counts_deltas[self.contribution_to_total_events_column_name] = (
                event_counts_deltas.apply(
                    lambda row: (
                        row[self.n_events_column_name] / row["events_norm"]
                        if row["events_norm"] > 0
                        else np.nan
                    ),
                    axis=1,
                )
            )
            event_counts_deltas[self.contribution_to_total_delta_column_name] = (
                event_counts_deltas.apply(
                    lambda row: (
                        row[self.total_score_delta_column_name] / row["delta_norm"]
                        if row["delta_norm"] > 0
                        else np.nan
                    ),
                    axis=1,
                )
            )
            # Accrual/improvement
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=norm_acc_imp,
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[
                self.contribution_to_total_accrual_events_column_name
            ] = event_counts_deltas.apply(
                lambda row: (
                    row[self.n_events_column_name] / row["acc_events_norm"]
                    if (row[self.event_type_column_name] != self.label_improvement)
                    and (row["acc_events_norm"] > 0)
                    else np.nan
                ),
                axis=1,
            )
            event_counts_deltas[
                self.contribution_to_total_accrual_delta_column_name
            ] = event_counts_deltas.apply(
                lambda row: (
                    row[self.total_score_delta_column_name] / row["acc_delta_norm"]
                    if (row[self.event_type_column_name] != self.label_improvement)
                    and (row["acc_delta_norm"] > 0)
                    else np.nan
                ),
                axis=1,
            )
            event_counts_deltas[
                self.contribution_to_total_improvement_events_column_name
            ] = event_counts_deltas.apply(
                lambda row: (
                    row[self.n_events_column_name] / row["imp_events_norm"]
                    if (row[self.event_type_column_name] == self.label_improvement)
                    and (row["imp_events_norm"] > 0)
                    else np.nan
                ),
                axis=1,
            )
            event_counts_deltas[
                self.contribution_to_total_improvement_delta_column_name
            ] = event_counts_deltas.apply(
                lambda row: (
                    row[self.total_score_delta_column_name] / row["imp_delta_norm"]
                    if (row[self.event_type_column_name] == self.label_improvement)
                    and (row["imp_delta_norm"] < 0)
                    else np.nan
                ),
                axis=1,
            )
            # Drop the norms.
            event_counts_deltas = event_counts_deltas.drop(
                columns=[
                    "events_norm",
                    "delta_norm",
                    "acc_events_norm",
                    "acc_delta_norm",
                    "imp_events_norm",
                    "imp_delta_norm",
                ]
            )

        # Accrual/improvement event type contribution
        if not get_stats_by_type:
            event_counts_deltas[
                self.contribution_of_accrual_to_total_events_column_name
            ] = event_counts_deltas.apply(
                lambda row: (
                    row[self.n_accrual_events_column_name]
                    / row[self.n_events_column_name]
                    if row[self.n_events_column_name] > 0
                    else np.nan
                ),
                axis=1,
            )
            event_counts_deltas[
                self.contribution_of_improvement_to_total_events_column_name
            ] = event_counts_deltas.apply(
                lambda row: (
                    row[self.n_improvement_events_column_name]
                    / row[self.n_events_column_name]
                    if row[self.n_events_column_name] > 0
                    else np.nan
                ),
                axis=1,
            )

        # Add number of follow-ups with events.
        # Helper function
        def _get_follow_ups_with_events(
            n_events_column_name, n_follow_ups_with_events_column_name
        ):
            return (
                stats_by_follow_up[stats_by_follow_up[n_events_column_name] > 0][
                    groupby_columns_with_types + [follow_up_id_column]
                ]
                .groupby(groupby_columns_with_types)
                .count()
                .reset_index()
                .rename(
                    columns={follow_up_id_column: n_follow_ups_with_events_column_name}
                )
            )

        # Overall
        follow_ups_with_events = _get_follow_ups_with_events(
            n_events_column_name=self.n_events_column_name,
            n_follow_ups_with_events_column_name=self.n_follow_ups_with_events_column_name,
        )
        event_counts_deltas = pd.merge(
            left=event_counts_deltas,
            right=follow_ups_with_events,
            on=groupby_columns_with_types,
            how="left",
        )
        event_counts_deltas[self.n_follow_ups_with_events_column_name] = (
            event_counts_deltas[self.n_follow_ups_with_events_column_name]
            .fillna(0)
            .astype(int)
        )
        event_counts_deltas[self.n_follow_ups_with_events_column_name + "_relative"] = (
            event_counts_deltas[self.n_follow_ups_with_events_column_name]
            / event_counts_deltas[self.n_follow_ups_column_name]
        )
        if not get_stats_by_type:
            # Accrual
            follow_ups_with_accrual_events = _get_follow_ups_with_events(
                n_events_column_name=self.n_accrual_events_column_name,
                n_follow_ups_with_events_column_name=self.n_follow_ups_with_accrual_events_column_name,
            )
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=follow_ups_with_accrual_events,
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[self.n_follow_ups_with_accrual_events_column_name] = (
                event_counts_deltas[self.n_follow_ups_with_accrual_events_column_name]
                .fillna(0)
                .astype(int)
            )
            # This division is safe, because we have > 0 follow-ups, duh...
            event_counts_deltas[
                self.n_follow_ups_with_accrual_events_column_name + "_relative"
            ] = (
                event_counts_deltas[self.n_follow_ups_with_accrual_events_column_name]
                / event_counts_deltas[self.n_follow_ups_column_name]
            )

            # Improvement
            follow_ups_with_improvement_events = _get_follow_ups_with_events(
                n_events_column_name=self.n_improvement_events_column_name,
                n_follow_ups_with_events_column_name=self.n_follow_ups_with_improvement_events_column_name,
            )
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=follow_ups_with_improvement_events,
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[
                self.n_follow_ups_with_improvement_events_column_name
            ] = (
                event_counts_deltas[
                    self.n_follow_ups_with_improvement_events_column_name
                ]
                .fillna(0)
                .astype(int)
            )
            event_counts_deltas[
                self.n_follow_ups_with_improvement_events_column_name + "_relative"
            ] = (
                event_counts_deltas[
                    self.n_follow_ups_with_improvement_events_column_name
                ]
                / event_counts_deltas[self.n_follow_ups_column_name]
            )

        # Follow-ups without events, with accrual only, with improvement only, with both
        # Without
        event_counts_deltas = pd.merge(
            left=event_counts_deltas,
            right=(
                stats_by_follow_up[stats_by_follow_up[self.n_events_column_name] == 0][
                    groupby_columns_with_types + [follow_up_id_column]
                ]
                .groupby(groupby_columns_with_types)
                .count()
                .reset_index()
                .rename(
                    columns={
                        follow_up_id_column: self.n_follow_ups_without_events_column_name
                    }
                )
            ),
            on=groupby_columns_with_types,
            how="left",
        )
        event_counts_deltas[self.n_follow_ups_without_events_column_name] = (
            event_counts_deltas[self.n_follow_ups_without_events_column_name]
            .fillna(0)
            .astype(int)
        )
        # Accrual only
        if not get_stats_by_type:
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=(
                    stats_by_follow_up[
                        (stats_by_follow_up[self.n_accrual_events_column_name] > 0)
                        & (
                            stats_by_follow_up[self.n_improvement_events_column_name]
                            == 0
                        )
                    ][groupby_columns + [follow_up_id_column]]
                    .groupby(groupby_columns)
                    .count()
                    .reset_index()
                    .rename(
                        columns={
                            follow_up_id_column: self.n_follow_ups_with_accrual_events_only_column_name
                        }
                    )
                ),
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[
                self.n_follow_ups_with_accrual_events_only_column_name
            ] = (
                event_counts_deltas[
                    self.n_follow_ups_with_accrual_events_only_column_name
                ]
                .fillna(0)
                .astype(int)
            )
            # Improvement only
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=(
                    stats_by_follow_up[
                        (stats_by_follow_up[self.n_accrual_events_column_name] == 0)
                        & (
                            stats_by_follow_up[self.n_improvement_events_column_name]
                            > 0
                        )
                    ][groupby_columns + [follow_up_id_column]]
                    .groupby(groupby_columns)
                    .count()
                    .reset_index()
                    .rename(
                        columns={
                            follow_up_id_column: self.n_follow_ups_with_improvement_events_only_column_name
                        }
                    )
                ),
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[
                self.n_follow_ups_with_improvement_events_only_column_name
            ] = (
                event_counts_deltas[
                    self.n_follow_ups_with_improvement_events_only_column_name
                ]
                .fillna(0)
                .astype(int)
            )
            # Both
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=(
                    stats_by_follow_up[
                        (stats_by_follow_up[self.n_accrual_events_column_name] > 0)
                        & (
                            stats_by_follow_up[self.n_improvement_events_column_name]
                            > 0
                        )
                    ][groupby_columns + [follow_up_id_column]]
                    .groupby(groupby_columns)
                    .count()
                    .reset_index()
                    .rename(
                        columns={
                            follow_up_id_column: self.n_follow_ups_with_accrual_and_improvement_events_column_name
                        }
                    )
                ),
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[
                self.n_follow_ups_with_accrual_and_improvement_events_column_name
            ] = (
                event_counts_deltas[
                    self.n_follow_ups_with_accrual_and_improvement_events_column_name
                ]
                .fillna(0)
                .astype(int)
            )

            # Follow-ups with acc == imp, acc > imp, acc < imp
            # Equal
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=(
                    stats_by_follow_up[
                        stats_by_follow_up[self.n_accrual_events_column_name]
                        == stats_by_follow_up[self.n_improvement_events_column_name]
                    ][groupby_columns + [follow_up_id_column]]
                    .groupby(groupby_columns)
                    .count()
                    .reset_index()
                    .rename(
                        columns={
                            follow_up_id_column: self.n_follow_ups_with_equal_n_accrual_improvement_column_name
                        }
                    )
                ),
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[
                self.n_follow_ups_with_equal_n_accrual_improvement_column_name
            ] = (
                event_counts_deltas[
                    self.n_follow_ups_with_equal_n_accrual_improvement_column_name
                ]
                .fillna(0)
                .astype(int)
            )
            # More accrual
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=(
                    stats_by_follow_up[
                        stats_by_follow_up[self.n_accrual_events_column_name]
                        > stats_by_follow_up[self.n_improvement_events_column_name]
                    ][groupby_columns + [follow_up_id_column]]
                    .groupby(groupby_columns)
                    .count()
                    .reset_index()
                    .rename(
                        columns={
                            follow_up_id_column: self.n_follow_ups_with_more_accrual_than_improvement_column_name
                        }
                    )
                ),
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[
                self.n_follow_ups_with_more_accrual_than_improvement_column_name
            ] = (
                event_counts_deltas[
                    self.n_follow_ups_with_more_accrual_than_improvement_column_name
                ]
                .fillna(0)
                .astype(int)
            )
            # More improvement
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=(
                    stats_by_follow_up[
                        stats_by_follow_up[self.n_accrual_events_column_name]
                        < stats_by_follow_up[self.n_improvement_events_column_name]
                    ][groupby_columns + [follow_up_id_column]]
                    .groupby(groupby_columns)
                    .count()
                    .reset_index()
                    .rename(
                        columns={
                            follow_up_id_column: self.n_follow_ups_with_less_accrual_than_improvement_column_name
                        }
                    )
                ),
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[
                self.n_follow_ups_with_less_accrual_than_improvement_column_name
            ] = (
                event_counts_deltas[
                    self.n_follow_ups_with_less_accrual_than_improvement_column_name
                ]
                .fillna(0)
                .astype(int)
            )

            # Follow-ups with delta 0, > 0, < 0
            # Delta 0
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=(
                    stats_by_follow_up[
                        stats_by_follow_up[self.total_score_delta_column_name] == 0
                    ][groupby_columns + [follow_up_id_column]]
                    .groupby(groupby_columns)
                    .count()
                    .reset_index()
                    .rename(
                        columns={
                            follow_up_id_column: self.n_follow_ups_delta_zero_column_name
                        }
                    )
                ),
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[self.n_follow_ups_delta_zero_column_name] = (
                event_counts_deltas[self.n_follow_ups_delta_zero_column_name]
                .fillna(0)
                .astype(int)
            )
            # Delta > 0
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=(
                    stats_by_follow_up[
                        stats_by_follow_up[self.total_score_delta_column_name] > 0
                    ][groupby_columns + [follow_up_id_column]]
                    .groupby(groupby_columns)
                    .count()
                    .reset_index()
                    .rename(
                        columns={
                            follow_up_id_column: self.n_follow_ups_positive_delta_column_name
                        }
                    )
                ),
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[self.n_follow_ups_positive_delta_column_name] = (
                event_counts_deltas[self.n_follow_ups_positive_delta_column_name]
                .fillna(0)
                .astype(int)
            )
            # Delta < 0
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=(
                    stats_by_follow_up[
                        stats_by_follow_up[self.total_score_delta_column_name] < 0
                    ][groupby_columns + [follow_up_id_column]]
                    .groupby(groupby_columns)
                    .count()
                    .reset_index()
                    .rename(
                        columns={
                            follow_up_id_column: self.n_follow_ups_negative_delta_column_name
                        }
                    )
                ),
                on=groupby_columns,
                how="left",
            )
            event_counts_deltas[self.n_follow_ups_negative_delta_column_name] = (
                event_counts_deltas[self.n_follow_ups_negative_delta_column_name]
                .fillna(0)
                .astype(int)
            )

        ### TIME TO EVENT ###

        # Time to first event survival analysis. Start with creating a
        # copy of the dataframe with the relevant columns.
        if get_stats_by_type:
            survival_stats_base = stats_by_follow_up[
                groupby_columns_with_types
                + [
                    follow_up_id_column,
                    self.first_timestamp_column_name,
                    self.duration_of_follow_up_column_name,
                    self.first_event_prefix + self.time_column_name,
                ]
            ].copy()
        else:
            survival_stats_base = stats_by_follow_up[
                groupby_columns
                + [
                    follow_up_id_column,
                    self.first_timestamp_column_name,
                    self.duration_of_follow_up_column_name,
                    self.first_event_prefix + self.time_column_name,
                    self.first_accrual_event_prefix + self.time_column_name,
                    self.first_improvement_event_prefix + self.time_column_name,
                ]
            ].copy()

        # Introduce observed yes/no and duration variables
        # as required by the lifelines package.
        # Overall
        survival_stats_base["observed"] = np.where(
            survival_stats_base[self.first_event_prefix + self.time_column_name] > 0,
            True,
            False,
        )
        survival_stats_base["time_to_event"] = (
            survival_stats_base[self.first_event_prefix + self.time_column_name]
            - survival_stats_base[self.first_timestamp_column_name]
        )
        survival_stats_base["duration"] = np.where(
            survival_stats_base["time_to_event"] > 0,
            survival_stats_base["time_to_event"],
            survival_stats_base[self.duration_of_follow_up_column_name],
        )
        if not get_stats_by_type:
            # Accrual
            survival_stats_base["accrual_observed"] = np.where(
                survival_stats_base[
                    self.first_accrual_event_prefix + self.time_column_name
                ]
                > 0,
                True,
                False,
            )
            survival_stats_base["accrual_time_to_event"] = (
                survival_stats_base[
                    self.first_accrual_event_prefix + self.time_column_name
                ]
                - survival_stats_base[self.first_timestamp_column_name]
            )
            survival_stats_base["accrual_duration"] = np.where(
                survival_stats_base["accrual_time_to_event"] > 0,
                survival_stats_base["accrual_time_to_event"],
                survival_stats_base[self.duration_of_follow_up_column_name],
            )
            # Improvement
            survival_stats_base["improvement_observed"] = np.where(
                survival_stats_base[
                    self.first_improvement_event_prefix + self.time_column_name
                ]
                > 0,
                True,
                False,
            )
            survival_stats_base["improvement_time_to_event"] = (
                survival_stats_base[
                    self.first_improvement_event_prefix + self.time_column_name
                ]
                - survival_stats_base[self.first_timestamp_column_name]
            )
            survival_stats_base["improvement_duration"] = np.where(
                survival_stats_base["improvement_time_to_event"] > 0,
                survival_stats_base["improvement_time_to_event"],
                survival_stats_base[self.duration_of_follow_up_column_name],
            )

        # TODO: Write a function for this...
        # We do the stats individually for each ID and type.
        survival_stats_list = []
        # The fastest (?) way to split the dataframe and to
        # get the stats for each ID and type combo is by
        # grouping, then writing the group elements to a
        # list, and then looping over this list.
        group_subdfs_list = [
            g for _, g in survival_stats_base.groupby(groupby_columns_with_types)
        ]
        for group_subdf in group_subdfs_list:
            # Get IDs of the group - we have to add them back
            # to the resulting stats 1-row dataframe later.
            # TODO: rewrite the survival stats function to
            # keep ID columns.
            subdf_ids = (
                group_subdf[groupby_columns_with_types]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            # Get survival stats
            group_subdf = get_median_survival_time_kaplan_meier(
                times_to_event_df=group_subdf,
                durations_column_name="duration",
                observed_column_name="observed",
                event_name="event",
            )
            # Add the IDs to the stats
            group_subdf = pd.concat([group_subdf, subdf_ids], axis=1)
            # Add the resulting stats 1-row dataframe to the results list
            survival_stats_list = survival_stats_list + [group_subdf]
        # Once the loop is complete, concatenate the dfs to one large df.
        # NOTE: concatenating a list comprehension is very inefficient,
        # thus this extra step with the list.
        survival_stats = pd.concat(survival_stats_list)
        # Remove columns from the survival stats function that are redundant.
        survival_stats = survival_stats.drop(columns=["n_events", "event_rate"])
        # Now add the results to the results dataframe.
        event_counts_deltas = pd.merge(
            left=event_counts_deltas,
            right=survival_stats,
            on=groupby_columns_with_types,
            how="left",
        )
        if not get_stats_by_type:
            # Accrual
            accrual_survival_stats_list = []
            # The fastest (?) way to split the dataframe and to
            # get the stats for each ID and type combo is by
            # grouping, then writing the group elements to a
            # list, and then looping over this list.
            group_subdfs_list = [
                g for _, g in survival_stats_base.groupby(groupby_columns)
            ]
            for group_subdf in group_subdfs_list:
                # Get IDs of the group - we have to add them back
                # to the resulting stats 1-row dataframe later.
                # TODO: rewrite the survival stats function to
                # keep ID columns.
                subdf_ids = (
                    group_subdf[groupby_columns]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                # Get survival stats
                group_subdf = get_median_survival_time_kaplan_meier(
                    times_to_event_df=group_subdf,
                    durations_column_name="accrual_duration",
                    observed_column_name="accrual_observed",
                    event_name="accrual_event",
                )
                # Add the IDs to the stats
                group_subdf = pd.concat([group_subdf, subdf_ids], axis=1)
                # Add the resulting stats 1-row dataframe to the results list
                accrual_survival_stats_list = accrual_survival_stats_list + [
                    group_subdf
                ]
            # Once the loop is complete, concatenate the dfs to one large df.
            # NOTE: concatenating a list comprehension is very inefficient,
            # thus this extra step with the list.
            accrual_survival_stats = pd.concat(accrual_survival_stats_list)
            # Remove columns from the survival stats function that are redundant.
            accrual_survival_stats = accrual_survival_stats.drop(
                columns=["n_accrual_events", "accrual_event_rate"]
            )
            # Now add the results to the results dataframe.
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=accrual_survival_stats,
                on=groupby_columns,
                how="left",
            )

            # Improvement
            improvement_survival_stats_list = []
            # The fastest (?) way to split the dataframe and to
            # get the stats for each ID and type combo is by
            # grouping, then writing the group elements to a
            # list, and then looping over this list.
            group_subdfs_list = [
                g for _, g in survival_stats_base.groupby(groupby_columns)
            ]
            for group_subdf in group_subdfs_list:
                # Get IDs of the group - we have to add them back
                # to the resulting stats 1-row dataframe later.
                # TODO: rewrite the survival stats function to
                # keep ID columns.
                subdf_ids = (
                    group_subdf[groupby_columns]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                # Get survival stats
                group_subdf = get_median_survival_time_kaplan_meier(
                    times_to_event_df=group_subdf,
                    durations_column_name="improvement_duration",
                    observed_column_name="improvement_observed",
                    event_name="improvement_event",
                )
                # Add the IDs to the stats
                group_subdf = pd.concat([group_subdf, subdf_ids], axis=1)
                # Add the resulting stats 1-row dataframe to the results list
                improvement_survival_stats_list = improvement_survival_stats_list + [
                    group_subdf
                ]
            # Once the loop is complete, concatenate the dfs to one large df.
            # NOTE: concatenating a list comprehension is very inefficient,
            # thus this extra step with the list.
            improvement_survival_stats = pd.concat(improvement_survival_stats_list)
            # Remove columns from the survival stats function that are redundant.
            improvement_survival_stats = improvement_survival_stats.drop(
                columns=["n_improvement_events", "improvement_event_rate"]
            )
            # Now add the results to the results dataframe.
            event_counts_deltas = pd.merge(
                left=event_counts_deltas,
                right=improvement_survival_stats,
                on=groupby_columns,
                how="left",
            )

        # Drop the dummy ID
        if self.dummy_id_column_name + "_groupby" in event_counts_deltas.columns:
            event_counts_deltas = event_counts_deltas.drop(
                columns=[self.dummy_id_column_name + "_groupby"]
            )

        return event_counts_deltas

    # Combination stats
    def get_combination_stats(
        self,
        stats_by_follow_up,
        follow_up_id_column,
        groupby_columns=None,
    ):
        """TBD, some thoughts:

        -   Compute how often a given combination of event types
            appears within a cohort (e.g. number of follow-ups with
            at least one PIRA and one RAW event, or number of follow-
            ups with undefined worsening only).

        -   Do this for all combinations, and with/without excluding
            other event types (e.g. follow-ups with at least one PIRA
            event and no events of any other types, and follow-ups with
            at least one PIRA events and any number of events of any
            other type). For 4 event types, this gives 28 combos.

        """
        # Groupby columns
        if groupby_columns == None:
            stats_by_follow_up[self.dummy_id_column_name + "_groupby"] = 0
            groupby_columns = [self.dummy_id_column_name + "_groupby"]
        # Get all type combinations
        event_types = [
            self.label_pira,
            self.label_pira_confirmed_in_raw_window,
            self.label_raw,
            self.label_undefined_worsening,
            self.label_improvement,
        ]
        type_combinations = list(
            chain(
                *[
                    list(combinations(event_types, r=i))
                    for i in range(1, len(event_types))
                ]
            )
        )

        # Get the queries. For example, the query for all follow-ups with
        # at least one PIRA event and no events of any other type reads
        # '(`PIRA` > 0) and (`PIRA confirmed in RAW window` == 0 and
        # `RAW` == 0 and `Undefined` == 0)', where the strings for the
        # event type names are taken from the type names given as class
        # args when instatianting the evaluation class.
        def _write_query(combination, selected_only):
            query_string = (
                # "(" + " and ".join(["(`" + elt + "` > 0)" for elt in combination]) + ")"
                " and ".join(["(`" + elt + "` > 0)" for elt in combination])
            )
            if selected_only:
                query_string = (
                    query_string
                    + " and "
                    # + "("
                    + " and ".join(
                        [
                            "(`" + elt + "`" + " == 0)"
                            for elt in event_types
                            if elt not in combination
                        ]
                    )
                    # + ")"
                )
            return query_string

        queries_selected_only = [
            _write_query(combination=type_combination, selected_only=True)
            for type_combination in type_combinations
        ]
        queries_inclusive = [
            _write_query(combination=type_combination, selected_only=False)
            for type_combination in type_combinations
        ]
        query_all = [" and ".join(["(`" + elt + "` > 0)" for elt in event_types])]
        queries = queries_selected_only + queries_inclusive + query_all

        # Get the norms - count follow-ups
        norm = (
            stats_by_follow_up[groupby_columns + [follow_up_id_column]]
            .drop_duplicates()
            .groupby(groupby_columns)
            .count()
            .reset_index()
        )
        # Count follow-ups with events; column will be dropped,
        # thus the sloppy naming.
        _follow_ups_with_events_column_name = "fups_with_events"
        norm_overall = (
            stats_by_follow_up[stats_by_follow_up[self.n_events_column_name] > 0][
                groupby_columns + [follow_up_id_column]
            ]
            .drop_duplicates()
            .groupby(groupby_columns)
            .count()
            .reset_index()
            .rename(columns={follow_up_id_column: _follow_ups_with_events_column_name})
        )

        # Pivot the counts - create a dataframe with one row per
        # follow-up and columns with event counts for each type.
        # The dataframe in this form can then be queried on row level.
        stats_by_follow_up_pivoted = (
            stats_by_follow_up[
                groupby_columns
                + [
                    follow_up_id_column,
                    self.event_type_column_name,
                    self.n_events_column_name,
                ]
            ]
            .pivot(
                index=groupby_columns + [follow_up_id_column],
                columns=self.event_type_column_name,
                values=self.n_events_column_name,
            )
            .reset_index()
        )

        # Helper function: query the dataframe with one of the
        # query strings, get the aggregated counts, and add the
        # query as row to the resulting df.
        def _get_counts_for_query(query_string):
            counts_df = (
                stats_by_follow_up_pivoted.query(query_string)[
                    groupby_columns + [follow_up_id_column]
                ]
                .groupby(groupby_columns)
                .count()
                .reset_index()
                .rename_axis(None, axis=1)
                .rename(columns={follow_up_id_column: self.n_follow_ups_column_name})
            )
            counts_df[self.combinations_query_column_name] = query_string
            return counts_df

        # Loop over combinations
        combination_counts_list = []
        for query_string in queries:
            combination_counts_for_query = _get_counts_for_query(
                query_string=query_string
            )
            combination_counts_list = combination_counts_list + [
                combination_counts_for_query
            ]

        # Concatenate
        combination_counts = pd.concat(combination_counts_list)

        # Not all combos have a follow-up, so we need a merge base.
        merge_base = pd.merge(
            left=norm,
            right=pd.DataFrame({self.combinations_query_column_name: queries}),
            how="cross",
        )
        merge_base = pd.merge(
            left=merge_base, right=norm_overall, on=groupby_columns, how="left"
        )
        merge_base[_follow_ups_with_events_column_name] = (
            merge_base[_follow_ups_with_events_column_name].fillna(0).astype(int)
        )
        combination_counts = pd.merge(
            left=merge_base,
            right=combination_counts,
            on=groupby_columns + [self.combinations_query_column_name],
            how="left",
        )
        combination_counts[self.n_follow_ups_column_name] = (
            combination_counts[self.n_follow_ups_column_name].fillna(0).astype(int)
        )

        # Relative
        combination_counts[self.n_follow_ups_column_name + "_relative"] = (
            combination_counts[self.n_follow_ups_column_name]
            / combination_counts[follow_up_id_column]
        )

        combination_counts[self.combinations_of_follow_ups_with_events_column_name] = (
            combination_counts[self.n_follow_ups_column_name]
            / combination_counts[_follow_ups_with_events_column_name]
        )

        # Drop norms
        combination_counts = combination_counts.drop(
            columns=[follow_up_id_column, _follow_ups_with_events_column_name]
        )

        # Drop the dummy ID
        if self.dummy_id_column_name + "_groupby" in combination_counts.columns:
            combination_counts = combination_counts.drop(
                columns=[self.dummy_id_column_name + "_groupby"]
            )

        return combination_counts


if __name__ == "__main__":
    pass
