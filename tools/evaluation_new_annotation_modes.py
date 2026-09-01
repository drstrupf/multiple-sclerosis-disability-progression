"""
This is a collection of tools for evaluating the output
of the EDSS worsening events annotation algorithm.

# Parts of the evaluation


## For each follow-up

### Overall
* Number of events
* Total delta EDSS
* Time to first event (+ type of the event, event score, event reference, delta of the event)

### By type
* Number of events by type
* Total delta EDSS by type
* Time to first event by type (+ event score, event reference, delta of the event)


## On cohort level

### Overall
* Number of events
* Total delta EDSS
* Median time to first event + 95%CI + distribution of types
* Median delta of first event + quantiles - or all events?

### By type
* Number of events by type
* Total delta EDSS by type
* Median time to first event + 95%CI by type
* Median delta of first event + quantiles by type - or all events?
* Combo counts

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
    label_undefined_progression: str = "Undefined"
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

    # combinations_query_column_name: str = "combination_query"
    # combinations_of_follow_ups_with_events_column_name: str = (
    #    "of_follow_ups_with_events"
    # )
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

    def get_merge_base(self, annotated_follow_ups, groupby_ids=None):
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
        return merge_base

    def get_events(self, annotated_follow_ups, get_accrual=True, get_improvement=True):
        """TBD, some thoughts:

        -   We get all accrual events by filtering on the progression
            flag column. Note that for merged events only the first
            event has this flag set to true. The events merged to this
            first event can be identified via the progression event ID.

        -   We also get the delta EDSS for each event. Note that in case
            of merged event the event score provided for the first event
            of a series of merged events (the one with the is progression
            event flag set to True) is the score for the entire merged
            event series, i.e. there is no sum over event IDs required.
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

    def get_follow_up_stats(self, annotated_follow_ups, id_columns=None):
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

        # Get the merge base
        merge_base = self.get_merge_base(
            annotated_follow_ups_copy,
            groupby_ids=id_columns,
        )

        # Select columns to group by.
        groupby_columns = id_columns

        # Get event dataframes
        all_events = self.get_events(
            annotated_follow_ups=annotated_follow_ups_copy,
            get_accrual=True,
            get_improvement=True,
        )
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
        stats_df[self.n_accrual_events_column_name] = stats_df[
            self.n_accrual_events_column_name
        ].astype(int)
        stats_df[self.n_improvement_events_column_name] = stats_df[
            self.n_improvement_events_column_name
        ].astype(int)

        ###### FIRST EVENTS ######
        # Select columns to aggregate for time to first event.
        # Keep track of the event type of the first event
        # (accrual or improvement)
        first_events_aggregation_columns = [
            self.time_column_name,
            self.event_reference_score_column_name,
            self.event_score_column_name,
            self.event_score_delta_column_name,
            self.event_type_column_name,
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
        stats_df = pd.merge(
            left=stats_df, right=first_events_accrual, on=groupby_columns, how="left"
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

        # Count follow-ups and get total number of event and total delta EDSS.
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
                total_improvement_events=(self.n_improvement_events_column_name, "sum"),
                total_delta=(self.total_score_delta_column_name, "sum"),
                total_accrual_delta=(self.total_accrual_score_delta_column_name, "sum"),
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
        # Accrual/improvement event type contribution
        event_counts_deltas[
            self.contribution_of_accrual_to_total_events_column_name
        ] = (
            event_counts_deltas[self.n_accrual_events_column_name]
            / event_counts_deltas[self.n_events_column_name]
        )
        event_counts_deltas[
            self.contribution_of_improvement_to_total_events_column_name
        ] = (
            event_counts_deltas[self.n_improvement_events_column_name]
            / event_counts_deltas[self.n_events_column_name]
        )

        # Add number of follow-ups with events.
        # Helper function

        def _get_follow_ups_with_events(
            n_events_column_name, n_follow_ups_with_events_column_name
        ):
            return (
                stats_by_follow_up[stats_by_follow_up[n_events_column_name] > 0][
                    groupby_columns + [follow_up_id_column]
                ]
                .groupby(groupby_columns)
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
            on=groupby_columns,
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
        event_counts_deltas[self.n_follow_ups_with_improvement_events_column_name] = (
            event_counts_deltas[self.n_follow_ups_with_improvement_events_column_name]
            .fillna(0)
            .astype(int)
        )
        event_counts_deltas[
            self.n_follow_ups_with_improvement_events_column_name + "_relative"
        ] = (
            event_counts_deltas[self.n_follow_ups_with_improvement_events_column_name]
            / event_counts_deltas[self.n_follow_ups_column_name]
        )

        # Follow-ups without events, with accrual only, with improvement only, with both
        # Without
        event_counts_deltas = pd.merge(
            left=event_counts_deltas,
            right=(
                stats_by_follow_up[stats_by_follow_up[self.n_events_column_name] == 0][
                    groupby_columns + [follow_up_id_column]
                ]
                .groupby(groupby_columns)
                .count()
                .reset_index()
                .rename(
                    columns={
                        follow_up_id_column: self.n_follow_ups_without_events_column_name
                    }
                )
            ),
            on=groupby_columns,
            how="left",
        )
        event_counts_deltas[self.n_follow_ups_without_events_column_name] = (
            event_counts_deltas[self.n_follow_ups_without_events_column_name]
            .fillna(0)
            .astype(int)
        )
        # Accrual only
        event_counts_deltas = pd.merge(
            left=event_counts_deltas,
            right=(
                stats_by_follow_up[
                    (stats_by_follow_up[self.n_accrual_events_column_name] > 0)
                    & (stats_by_follow_up[self.n_improvement_events_column_name] == 0)
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
        event_counts_deltas[self.n_follow_ups_with_accrual_events_only_column_name] = (
            event_counts_deltas[self.n_follow_ups_with_accrual_events_only_column_name]
            .fillna(0)
            .astype(int)
        )
        # Improvement only
        event_counts_deltas = pd.merge(
            left=event_counts_deltas,
            right=(
                stats_by_follow_up[
                    (stats_by_follow_up[self.n_accrual_events_column_name] == 0)
                    & (stats_by_follow_up[self.n_improvement_events_column_name] > 0)
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
                    & (stats_by_follow_up[self.n_improvement_events_column_name] > 0)
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
        # Accrual
        survival_stats_base["accrual_observed"] = np.where(
            survival_stats_base[self.first_accrual_event_prefix + self.time_column_name]
            > 0,
            True,
            False,
        )
        survival_stats_base["accrual_time_to_event"] = (
            survival_stats_base[self.first_accrual_event_prefix + self.time_column_name]
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
        group_subdfs_list = [g for _, g in survival_stats_base.groupby(groupby_columns)]
        for group_subdf in group_subdfs_list:
            # Get IDs of the group - we have to add them back
            # to the resulting stats 1-row dataframe later.
            # TODO: rewrite the survival stats function to
            # keep ID columns.
            subdf_ids = (
                group_subdf[groupby_columns].drop_duplicates().reset_index(drop=True)
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
            on=groupby_columns,
            how="left",
        )

        # Accrual
        accrual_survival_stats_list = []
        # The fastest (?) way to split the dataframe and to
        # get the stats for each ID and type combo is by
        # grouping, then writing the group elements to a
        # list, and then looping over this list.
        group_subdfs_list = [g for _, g in survival_stats_base.groupby(groupby_columns)]
        for group_subdf in group_subdfs_list:
            # Get IDs of the group - we have to add them back
            # to the resulting stats 1-row dataframe later.
            # TODO: rewrite the survival stats function to
            # keep ID columns.
            subdf_ids = (
                group_subdf[groupby_columns].drop_duplicates().reset_index(drop=True)
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
            accrual_survival_stats_list = accrual_survival_stats_list + [group_subdf]
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
        group_subdfs_list = [g for _, g in survival_stats_base.groupby(groupby_columns)]
        for group_subdf in group_subdfs_list:
            # Get IDs of the group - we have to add them back
            # to the resulting stats 1-row dataframe later.
            # TODO: rewrite the survival stats function to
            # keep ID columns.
            subdf_ids = (
                group_subdf[groupby_columns].drop_duplicates().reset_index(drop=True)
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

    # Stats on merged events by follow-up
    def get_follow_up_merged_events_stats(
        self,
        annotated_follow_ups,
        follow_up_id_column=None,
        groupby_columns=None,
    ):
        """TBD"""
        # Follow-up ID, add dummy if not provided
        if follow_up_id_column is None:
            annotated_follow_ups[self.dummy_id_column_name] = 0
            follow_up_id_column = self.dummy_id_column_name
        # Groupby columns, add dummy if not provided
        if groupby_columns == None:
            annotated_follow_ups[self.dummy_id_column_name + "_groupby"] = 0
            groupby_columns = [self.dummy_id_column_name + "_groupby"]

        # Since stats are by type, we need a separate info on
        # event type by event ID, because the type is only
        # provided for the first entry per merged event.
        event_types_info = annotated_follow_ups[
            (~annotated_follow_ups[self.event_id_column_name].isna())
            & (~annotated_follow_ups[self.event_type_column_name].isna())
        ][
            groupby_columns
            + [
                follow_up_id_column,
                self.event_id_column_name,
                self.event_type_column_name,
            ]
        ]
        # Now count how many assessments are merged for each event...
        merged_counts = (
            annotated_follow_ups[
                ~annotated_follow_ups[self.event_id_column_name].isna()
            ][
                groupby_columns
                + [
                    follow_up_id_column,
                    self.time_column_name,
                    self.event_id_column_name,
                ]
            ]
            .groupby(groupby_columns + [follow_up_id_column, self.event_id_column_name])
            .count()
            .reset_index()
            .rename(
                columns={self.time_column_name: self.n_merged_assessments_column_name}
            )
        )
        # Add type info
        merged_counts = pd.merge(
            left=merged_counts,
            right=event_types_info,
            on=groupby_columns + [follow_up_id_column, self.event_id_column_name],
            how="left",
        )

        ## ...then group by number of assessments merged and count events.
        groupby_for_count = groupby_columns + [
            follow_up_id_column,
            self.n_merged_assessments_column_name,
            self.event_type_column_name,
        ]
        merged_counts = (
            merged_counts.groupby(groupby_for_count)
            .count()
            .reset_index()
            .rename(columns={self.event_id_column_name: self.n_events_column_name})
        )

        # Drop the dummy IDs if applicable
        if self.dummy_id_column_name in merged_counts.columns:
            merged_counts = merged_counts.drop(columns=[self.dummy_id_column_name])
        if self.dummy_id_column_name + "_groupby" in merged_counts.columns:
            merged_counts = merged_counts.drop(
                columns=[self.dummy_id_column_name + "_groupby"]
            )

        return merged_counts

    # TODO: HERE
    def get_cohort_merged_events_stats(
        self,
        merged_event_stats_by_follow_up,
        groupby_columns=None,
    ):
        """TBD"""
        # Groupby columns, add dummy if not provided
        if groupby_columns == None:
            merged_event_stats_by_follow_up[self.dummy_id_column_name + "_groupby"] = 0
            groupby_columns = [self.dummy_id_column_name + "_groupby"]
        # Groupby columns for overall counts
        groupby_for_count = groupby_columns + [
            self.n_merged_assessments_column_name,
            self.event_type_column_name,
        ]
        return (
            merged_event_stats_by_follow_up[
                groupby_for_count + [self.n_events_column_name]
            ]
            .groupby(groupby_for_count)
            .sum()
            .reset_index()
        )


if __name__ == "__main__":
    pass
