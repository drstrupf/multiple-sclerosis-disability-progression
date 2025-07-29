import numpy as np
import pandas as pd
import lifelines
from dataclasses import dataclass
from itertools import chain, combinations

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


# Survival for right-censored data, Kaplan-Meier
def get_median_survival_time_kaplan_meier(
    times_to_event_df, durations_column_name="duration", observed_column_name="observed"
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
                "n_events": n_events_observed,
                "event_rate": event_rate,
                "median_time_to_first_progression": median_time_to_event,
                "median_time_to_first_progression_lower_95CI": median_time_to_event_ci.loc[
                    0.5
                ][
                    "KM_estimate_lower_0.95"
                ],
                "median_time_to_first_progression_upper_95CI": median_time_to_event_ci.loc[
                    0.5
                ][
                    "KM_estimate_upper_0.95"
                ],
            }
        ]
    )


@dataclass
class EDSSProgressionEvaluation:
    # Input
    edss_score_column_name: str = "edss_score"
    time_column_name: str = "days_after_baseline"
    is_progression_flag_column_name: str = "is_progression"
    progression_type_column_name: str = "progression_type"
    progression_score_column_name: str = "progression_score"
    progression_reference_score_column_name: str = "progression_reference_score"
    progression_event_id_column_name: str = "progression_event_id"
    label_undefined_progression: str = "Undefined"
    label_pira: str = "PIRA"
    label_pira_confirmed_in_raw_window: str = "PIRA with relapse during confirmation"
    label_raw: str = "RAW"
    # Output
    progression_score_delta_column_name: str = "progression_score_delta"
    n_accrual_events_column_name: str = "total_events"
    total_accrual_events_edss_delta: str = "total_delta"
    first_event_prefix: str = "first_event_"
    first_timestamp_column_name: str = "first_timestamp"
    last_timestamp_column_name: str = "last_timestamp"
    duration_of_follow_up_column_name: str = "duration_of_follow_up"
    n_follow_ups_column_name: str = "n_follow_ups"
    n_follow_ups_with_events_column_name: str = "n_follow_ups_with_events"
    contribution_to_total_events_column_name: str = "contribution_to_total_events"
    contribution_to_total_delta_column_name: str = "contribution_to_total_delta"
    combinations_query_column_name: str = "combination_query"
    combinations_of_follow_ups_with_events_column_name: str = (
        "of_follow_ups_with_events"
    )
    n_merged_assessments_column_name: str = "n_merged_assessments"
    # Helpers
    dummy_id_column_name: str = "dummy_id"
    """TBD"""

    def __post_init__(self):
        """Non-boilerplate __init__ part."""
        pass

    def get_merge_base(
        self, annotated_follow_ups, get_stats_by_type=True, groupby_ids=[]
    ):
        """TBD, some thoughts:

        -   Evaluation uses groupby/aggregate operations. If e.g. an event
            type is not present in the data, then it does not appear in the
            aggregation results. Or a follow-up with zero events would drop
            out when aggregating. Therefore, we first construct a dataframe
            with all follow-ups and event types that we can use as a left
            merge base for collecting aggregation results.

        -   This should also work for a single follow-up that satisfies the
            minimum column requirement of the annotation algorithm (timestamp
            and EDSS score columns available), thus it creates a dummy ID if
            no groupby IDs are provided when calling the method.

        -   Add first and last timestamp and duration of follow-up to this
            merge base (required for time to event etc.).
        """
        # We need at least one ID to group by; introduce dummy ID
        # if none provided.
        if groupby_ids == []:
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
                        self.progression_type_column_name: [
                            self.label_pira,
                            self.label_pira_confirmed_in_raw_window,
                            self.label_raw,
                            self.label_undefined_progression,
                        ]
                    }
                ),
                how="cross",
            )
        return merge_base

    def get_accrual_events(self, annotated_follow_ups):
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
        accrual_events = annotated_follow_ups[
            annotated_follow_ups[self.is_progression_flag_column_name]
        ].copy()
        accrual_events[self.progression_score_delta_column_name] = (
            accrual_events[self.progression_score_column_name]
            - accrual_events[self.progression_reference_score_column_name]
        )
        return accrual_events

    def get_follow_up_stats(
        self, annotated_follow_ups, get_stats_by_type=True, id_columns=[]
    ):
        """TBD, some thoughts:

        -   We want to know the count of events, the distribution of
            event types, and the time to first event etc. for each
            individual follow-up.
        """
        # Work on copy
        annotated_follow_ups_copy = annotated_follow_ups.copy()
        # Groupby/agg operations drop groups with zero elements,
        # thus we first create a merge base with all follow ups
        # to keep track of them.
        if id_columns == []:
            annotated_follow_ups_copy[self.dummy_id_column_name] = 0
            id_columns = [self.dummy_id_column_name]
        merge_base = self.get_merge_base(
            annotated_follow_ups_copy,
            get_stats_by_type=get_stats_by_type,
            groupby_ids=id_columns,
        )

        # Select columns to group by.
        groupby_columns = id_columns
        if get_stats_by_type:
            groupby_columns = groupby_columns + [self.progression_type_column_name]

        # Select columns to aggregate for counts and deltas.
        counts_deltas_aggregation_columns = [
            self.progression_event_id_column_name,
            self.progression_score_delta_column_name,
        ]

        # Select columns to aggregate for time to first event.
        first_events_aggregation_columns = [
            self.time_column_name,
            self.progression_reference_score_column_name,
            self.progression_score_column_name,
            self.progression_score_delta_column_name,
        ]

        # If we compute overall stats, keep track of the event
        # type of the first disability accrual event.
        if not get_stats_by_type:
            first_events_aggregation_columns = first_events_aggregation_columns + [
                self.progression_type_column_name
            ]

        # Get a dataframe with accrual events only - everything
        # required for the stats is in these rows.
        accrual_events = self.get_accrual_events(
            annotated_follow_ups=annotated_follow_ups_copy
        )

        # Count events and sum delta EDSS.
        event_counts_deltas = (
            accrual_events[groupby_columns + counts_deltas_aggregation_columns]
            .groupby(groupby_columns)
            .agg(
                count=(self.progression_event_id_column_name, "count"),
                total_delta=(self.progression_score_delta_column_name, "sum"),
            )
            .reset_index()
            .rename(
                columns={
                    "count": self.n_accrual_events_column_name,
                    "total_delta": self.total_accrual_events_edss_delta,
                }
            )
        )
        # Merge dataframes and adjust data types.
        stats_df = pd.merge(
            left=merge_base, right=event_counts_deltas, on=groupby_columns, how="left"
        ).fillna(0)
        stats_df[self.n_accrual_events_column_name] = stats_df[
            self.n_accrual_events_column_name
        ].astype(int)

        # Time to first event - get the first event per group
        # and extract its stats.
        first_events = accrual_events.loc[
            accrual_events.groupby(groupby_columns)[self.time_column_name].idxmin()
        ][groupby_columns + first_events_aggregation_columns].rename(
            columns={
                column_name: self.first_event_prefix + column_name
                for column_name in first_events_aggregation_columns
            }
        )
        # Merge dataframes - leave NaNs where there are no events!
        # NOTE: The NaNs are important for survival analysis later
        # on - they indicate follow-ups without event during the
        # observation period.
        stats_df = pd.merge(
            left=stats_df, right=first_events, on=groupby_columns, how="left"
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
        groupby_columns=[],
    ):
        """TBD, some thoughts:

        -   Get stats on event counts, event rates, type distribution,
            and time  to event stats on cohort level.

        -   Builds on stats per follow-up, i.e. the output of the
            get_follow_up_stats method above. If stats are by type,
            every follow-up gets a row for each type irrespective of
            whether there are events of this type. This makes groupby
            and aggregation safe.

        -   Allows only one ID column for each follow-up, since it
            aggregates over follow-ups. This ID column will be used
            to count follow-ups.

        -   Groupby columns allow processing multiple cohort at once
            by identifying each cohort with one ore more IDs. In our
            case, the groupby column is the parameter combo ID.

        """
        # Follow-up ID, add dummy if not provided
        if follow_up_id_column is None:
            stats_by_follow_up[self.dummy_id_column_name] = 0
            follow_up_id_column = self.dummy_id_column_name
        # Groupby columns
        if groupby_columns == []:
            stats_by_follow_up[self.dummy_id_column_name + "_groupby"] = 0
            groupby_columns = [self.dummy_id_column_name + "_groupby"]
        # Introduce new groupby list that includes types if applicable,
        # while keeping the original list, too, for convenience. We need
        # the groupby list without types later for computing the norms.
        groupby_columns_with_types = groupby_columns
        if get_stats_by_type:
            groupby_columns_with_types = groupby_columns_with_types + [
                self.progression_type_column_name
            ]

        # Count follow-ups and get total number of event and total delta EDSS.
        event_counts_deltas = (
            stats_by_follow_up[
                groupby_columns_with_types
                + [
                    follow_up_id_column,
                    self.n_accrual_events_column_name,
                    self.total_accrual_events_edss_delta,
                ]
            ]
            .groupby(groupby_columns_with_types)
            .agg(
                n_follow_ups=(follow_up_id_column, "count"),
                total_events=(self.n_accrual_events_column_name, "sum"),
                total_delta=(self.total_accrual_events_edss_delta, "sum"),
            )
            .reset_index()
            .rename(
                columns={
                    "n_follow_ups": self.n_follow_ups_column_name,
                    "total_events": self.n_accrual_events_column_name,
                    "total_delta": self.total_accrual_events_edss_delta,
                }
            )
        )

        # Add number of follow-ups with events.
        follow_ups_with_events = (
            stats_by_follow_up[
                stats_by_follow_up[self.n_accrual_events_column_name] > 0
            ][groupby_columns_with_types + [follow_up_id_column]]
            .groupby(groupby_columns_with_types)
            .count()
            .reset_index()
            .rename(
                columns={follow_up_id_column: self.n_follow_ups_with_events_column_name}
            )
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

        # Compute event rate (follow-ups with events over total number of follow-ups).
        event_counts_deltas[self.n_follow_ups_with_events_column_name + "_relative"] = (
            event_counts_deltas[self.n_follow_ups_with_events_column_name]
            / event_counts_deltas[self.n_follow_ups_column_name]
        )

        # If by type, get type distributions (i.e. the proportion of events that
        # are of a given type).
        if get_stats_by_type:
            # Count total events and deltas irrespective of type for the norm.
            norm = (
                stats_by_follow_up[
                    groupby_columns
                    + [
                        self.n_accrual_events_column_name,
                        self.total_accrual_events_edss_delta,
                    ]
                ]
                .groupby(groupby_columns)
                .agg(
                    events_norm=(self.n_accrual_events_column_name, "sum"),
                    delta_norm=(self.total_accrual_events_edss_delta, "sum"),
                )
                .reset_index()
            )
            event_counts_deltas = pd.merge(
                left=event_counts_deltas, right=norm, on=groupby_columns, how="left"
            )
            event_counts_deltas[self.contribution_to_total_events_column_name] = (
                event_counts_deltas[self.n_accrual_events_column_name]
                / event_counts_deltas["events_norm"]
            )
            event_counts_deltas[self.contribution_to_total_delta_column_name] = (
                event_counts_deltas[self.total_accrual_events_edss_delta]
                / event_counts_deltas["delta_norm"]
            )
            # Drop the norms.
            event_counts_deltas = event_counts_deltas.drop(
                columns=["events_norm", "delta_norm"]
            )

        # Time to first event survival analysis. Start with creating a
        # copy of the dataframe with the relevant columns.
        survival_stats_base = stats_by_follow_up[
            groupby_columns_with_types
            + [
                follow_up_id_column,
                self.first_timestamp_column_name,
                self.duration_of_follow_up_column_name,
                self.first_event_prefix + self.time_column_name,
            ]
        ].copy()

        # Introduce observed yes/no and duration variables
        # as required by the lifelines package.
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
        groupby_columns=[],
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
        if groupby_columns == []:
            stats_by_follow_up[self.dummy_id_column_name + "_groupby"] = 0
            groupby_columns = [self.dummy_id_column_name + "_groupby"]
        # Get all type combinations
        progression_types = [
            self.label_pira,
            self.label_pira_confirmed_in_raw_window,
            self.label_raw,
            self.label_undefined_progression,
        ]
        type_combinations = list(
            chain(
                *[
                    list(combinations(progression_types, r=i))
                    for i in range(1, len(progression_types))
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
                "(" + " and ".join(["`" + elt + "` > 0" for elt in combination]) + ")"
            )
            if selected_only:
                query_string = (
                    query_string
                    + " and "
                    + "("
                    + " and ".join(
                        [
                            "`" + elt + "`" + " == 0"
                            for elt in progression_types
                            if elt not in combination
                        ]
                    )
                    + ")"
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
        queries = queries_selected_only + queries_inclusive

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
            stats_by_follow_up[
                stats_by_follow_up[self.n_accrual_events_column_name] > 0
            ][groupby_columns + [follow_up_id_column]]
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
                    self.progression_type_column_name,
                    self.n_accrual_events_column_name,
                ]
            ]
            .pivot(
                index=groupby_columns + [follow_up_id_column],
                columns=self.progression_type_column_name,
                values=self.n_accrual_events_column_name,
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

    # Stats on merged events by follow-up
    def get_follow_up_merged_events_stats(
        self,
        annotated_follow_ups,
        follow_up_id_column=None,
        get_stats_by_type=True,
        groupby_columns=[],
    ):
        """TBD, some thoughts:

        -   This function is for one follow-up only. See
            below for a function for merged stats on cohort
            level (builds on the output of this one).

        -   Merged events can be identified via event ID.
            Only the first assessment in a merged event has
            the is event flag set to True.

        -   It is not straightforward to define 'individual
            event' in the context of merging; e.g. repetition
            measurements have the same score as the previous
            one and would not be an event after post-event re-
            baselining after the first, but they would be an
            event w.r.t. the reference before the first event.
            Also, just counting the number of different EDSS
            scores within a series of merged events does not
            necessarily yield something useful, since confirmed
            event scores can be lower than the actual measured
            score. Thus, here the number of events merged into
            one event is simply defined as the number of EDSS
            assessments that fall into a merged event series.
            To get the number of assessment, we can count the
            timestamps (duplicate timestamps are not allowed
            by the progression annotation algorithm, so this
            is safe).

        """
        # Only keep event types with events when getting the
        # merged event counts by type.

        # Follow-up ID, add dummy if not provided
        if follow_up_id_column is None:
            annotated_follow_ups[self.dummy_id_column_name] = 0
            follow_up_id_column = self.dummy_id_column_name
        # Groupby columns, add dummy if not provided
        if groupby_columns == []:
            annotated_follow_ups[self.dummy_id_column_name + "_groupby"] = 0
            groupby_columns = [self.dummy_id_column_name + "_groupby"]

        # If stats are by type, we need a separate info on
        # progression type by progression event ID, because
        # the type is only provided for the first entry per
        # merged event.
        if get_stats_by_type:
            progression_types_info = annotated_follow_ups[
                (~annotated_follow_ups[self.progression_event_id_column_name].isna())
                & (~annotated_follow_ups[self.progression_type_column_name].isna())
            ][
                groupby_columns
                + [
                    follow_up_id_column,
                    self.progression_event_id_column_name,
                    self.progression_type_column_name,
                ]
            ]

        # Now count how many assessments are merged for each event...
        merged_counts = (
            annotated_follow_ups[
                ~annotated_follow_ups[self.progression_event_id_column_name].isna()
            ][
                groupby_columns
                + [
                    follow_up_id_column,
                    self.time_column_name,
                    self.progression_event_id_column_name,
                ]
            ]
            .groupby(
                groupby_columns
                + [follow_up_id_column, self.progression_event_id_column_name]
            )
            .count()
            .reset_index()
            .rename(
                columns={self.time_column_name: self.n_merged_assessments_column_name}
            )
        )
        if get_stats_by_type:
            merged_counts = pd.merge(
                left=merged_counts,
                right=progression_types_info,
                on=groupby_columns
                + [follow_up_id_column, self.progression_event_id_column_name],
                how="left",
            )

        # ...then group by number of assessments merged and count events.
        groupby_for_count = groupby_columns + [
            follow_up_id_column,
            self.n_merged_assessments_column_name,
        ]
        if get_stats_by_type:
            groupby_for_count = groupby_for_count + [self.progression_type_column_name]
        merged_counts = (
            merged_counts.groupby(groupby_for_count)
            .count()
            .reset_index()
            .rename(
                columns={
                    self.progression_event_id_column_name: self.n_accrual_events_column_name
                }
            )
        )

        # Drop the dummy IDs if applicable
        if self.dummy_id_column_name in merged_counts.columns:
            merged_counts = merged_counts.drop(columns=[self.dummy_id_column_name])
        if self.dummy_id_column_name + "_groupby" in merged_counts.columns:
            merged_counts = merged_counts.drop(
                columns=[self.dummy_id_column_name + "_groupby"]
            )

        return merged_counts

    def get_cohort_merged_events_stats(
        self,
        merged_event_stats_by_follow_up,
        get_stats_by_type=True,
        groupby_columns=[],
    ):
        """TBD, some thoughts:

        -   Merged stats on cohort level. Requires merged stats
            on follow-up level as input. See description of the
            function for follow-up level stats for details on how
            the number of merged events is defined.
        """
        # Groupby columns, add dummy if not provided
        if groupby_columns == []:
            merged_event_stats_by_follow_up[self.dummy_id_column_name + "_groupby"] = 0
            groupby_columns = [self.dummy_id_column_name + "_groupby"]
        # Groupby columns for overall counts
        groupby_for_count = groupby_columns + [
            self.n_merged_assessments_column_name,
        ]
        if get_stats_by_type:
            groupby_for_count = groupby_for_count + [self.progression_type_column_name]

        return (
            merged_event_stats_by_follow_up[
                groupby_for_count + [self.n_accrual_events_column_name]
            ]
            .groupby(groupby_for_count)
            .sum()
            .reset_index()
        )


if __name__ == "__main__":
    pass
