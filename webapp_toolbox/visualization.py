import numpy as np
import pandas as pd
import streamlit as st
import lifelines

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
sns.set_style("whitegrid", {"grid.color": "gainsboro"})


def plot_single_annotated_followup(
    annotated_example_follow_up_df,
    edss_score_column_name="edss_score",
    time_column_name="days_after_baseline",
    first_progression_flag_column_name="is_first_progression",
    figsize=(12, 8),
):
    # Some global formatting options that could be promoted to arg sometime...
    BASE_PLOT_COLOR = "black"
    BASE_PLOT_MARKER = "o"
    BASE_PLOT_LINEWIDTH = 1
    MARKERSIZE = 8
    OVERLAY_COLOR = "red"
    OVERLAY_MARKER = "D"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    g = sns.lineplot(
        data=annotated_example_follow_up_df,
        x=time_column_name,
        y=edss_score_column_name,
        color=BASE_PLOT_COLOR,
        marker=BASE_PLOT_MARKER,
        linewidth=BASE_PLOT_LINEWIDTH,
        markersize=MARKERSIZE,
        legend=False,
        ax=ax,
    )
    # if options_submitted:
    first_progression = annotated_example_follow_up_df[
        annotated_example_follow_up_df[first_progression_flag_column_name]
    ]
    if len(first_progression) == 1:
        g_overlay = sns.lineplot(
            data=first_progression,
            x=time_column_name,
            y=edss_score_column_name,
            color=OVERLAY_COLOR,
            marker=OVERLAY_MARKER,
            markersize=MARKERSIZE,
            legend=False,
            ax=ax,
        )

    ax.legend(
        title="Legend",
        labels=[
            "EDSS assessment",
            "First progression event",
        ],
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=BASE_PLOT_COLOR,
                marker=BASE_PLOT_MARKER,
                linewidth=0,
                markersize=MARKERSIZE * 0.75,
            ),
            plt.Line2D(
                [0],
                [0],
                color=OVERLAY_COLOR,
                marker=OVERLAY_MARKER,
                linewidth=0,
                markersize=MARKERSIZE * 0.75,
            ),
        ],
    )

    ax.set_xlabel("Time after baseline")
    ax.set_xticks(annotated_example_follow_up_df[time_column_name])
    ax.set_ylabel("EDSS score")
    g.set_ylim((-0.1, 10.1))
    g.set_yticks([i * 0.5 for i in range(21)])

    sns.despine(bottom=True, left=True, right=True, top=True)

    st.pyplot(fig, clear_figure=True)


def plot_single_kaplan_meier(
    times_to_event_df,
    durations_column_name="duration",
    observed_column_name="observed",
    figsize=(12, 8),
    xlim=None,
):
    # Some global formatting options that could be promoted to arg sometime...
    BASE_PLOT_COLOR = "indigo"
    BASE_PLOT_LINEWIDTH = 1

    fig, ax = plt.subplots(1, 1, figsize=figsize)

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
    median_time_to_event_ci_lower = median_time_to_event_ci.loc[0.5][
        "KM_estimate_lower_0.95"
    ]

    median_time_to_event_ci_upper = median_time_to_event_ci.loc[0.5][
        "KM_estimate_upper_0.95"
    ]

    kaplan_meier_fitter.plot_survival_function(
        color=BASE_PLOT_COLOR, linewidth=BASE_PLOT_LINEWIDTH
    )

    if median_time_to_event >= 100_000:
        median_time_to_event_str = r"n/a"
    else:
        median_time_to_event_str = str(int(np.ceil(median_time_to_event)))
    if median_time_to_event_ci_lower >= 100_000:
        median_time_to_event_ci_lower_str = r"n/a"
    else:
        median_time_to_event_ci_lower_str = str(
            int(np.ceil(median_time_to_event_ci_lower))
        )
    if median_time_to_event_ci_upper >= 100_000:
        median_time_to_event_ci_upper_str = r"n/a"
    else:
        median_time_to_event_ci_upper_str = str(
            int(np.ceil(median_time_to_event_ci_upper))
        )

    ax.legend(
        title="Legend",
        labels=[
            "Kaplan-Meier estimate",
        ],
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=BASE_PLOT_COLOR,
                linewidth=BASE_PLOT_LINEWIDTH,
            ),
        ],
    )

    ax.set_xlim(xlim)
    ax.set_ylim((0, 1))

    ax.set_xlabel("Days after baseline")

    ax.set_title(
        "Event rate: "
        + str(np.round(event_rate * 100, 2))
        + "%"
        + "\nMedian time to first progression (95%CI): "
        + median_time_to_event_str
        + " days ("
        + median_time_to_event_ci_lower_str
        + " - "
        + median_time_to_event_ci_upper_str
        + " days)"
    )

    sns.despine(bottom=True, left=True, right=True, top=True)

    st.pyplot(fig, clear_figure=True)


def plot_kaplan_meier_comparison(
    times_to_event_df_1,
    times_to_event_df_2,
    durations_column_name="duration",
    observed_column_name="observed",
    figsize=(12, 8),
    xlim=None,
):
    # Some global formatting options that could be promoted to arg sometime...
    BASE_PLOT_COLOR = "black"
    BASE_PLOT_LINEWIDTH = 1
    OVERLAY_PLOT_COLOR = "indigo"
    OVERLAY_PLOT_LINEWIDTH = 1

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    kaplan_meier_fitter_1 = lifelines.KaplanMeierFitter()
    durations_1 = times_to_event_df_1[durations_column_name]
    observed_1 = times_to_event_df_1[observed_column_name]
    kaplan_meier_fitter_1.fit(
        durations=durations_1,
        event_observed=observed_1,
    )

    kaplan_meier_fitter_2 = lifelines.KaplanMeierFitter()
    durations_2 = times_to_event_df_2[durations_column_name]
    observed_2 = times_to_event_df_2[observed_column_name]
    kaplan_meier_fitter_2.fit(
        durations=durations_2,
        event_observed=observed_2,
    )

    # Count events, absolute and relative
    n_observations_1 = len(durations_1)
    n_events_observed_1 = sum(observed_1)
    event_rate_1 = n_events_observed_1 / n_observations_1

    n_observations_2 = len(durations_2)
    n_events_observed_2 = sum(observed_2)
    event_rate_2 = n_events_observed_2 / n_observations_2

    # Get median survival times with 95% CI
    median_time_to_event_1 = kaplan_meier_fitter_1.median_survival_time_
    median_time_to_event_ci_1 = lifelines.utils.median_survival_times(
        kaplan_meier_fitter_1.confidence_interval_
    )
    median_time_to_event_ci_lower_1 = median_time_to_event_ci_1.loc[0.5][
        "KM_estimate_lower_0.95"
    ]

    median_time_to_event_ci_upper_1 = median_time_to_event_ci_1.loc[0.5][
        "KM_estimate_upper_0.95"
    ]

    median_time_to_event_2 = kaplan_meier_fitter_2.median_survival_time_
    median_time_to_event_ci_2 = lifelines.utils.median_survival_times(
        kaplan_meier_fitter_2.confidence_interval_
    )
    median_time_to_event_ci_lower_2 = median_time_to_event_ci_2.loc[0.5][
        "KM_estimate_lower_0.95"
    ]

    median_time_to_event_ci_upper_2 = median_time_to_event_ci_2.loc[0.5][
        "KM_estimate_upper_0.95"
    ]

    # Plot the curves
    kaplan_meier_fitter_1.plot_survival_function(
        color=BASE_PLOT_COLOR, linewidth=BASE_PLOT_LINEWIDTH
    )
    kaplan_meier_fitter_2.plot_survival_function(
        color=OVERLAY_PLOT_COLOR, linewidth=OVERLAY_PLOT_LINEWIDTH
    )

    if median_time_to_event_1 >= 100_000:
        median_time_to_event_str_1 = r"n/a"
    else:
        median_time_to_event_str_1 = str(int(np.ceil(median_time_to_event_1)))
    if median_time_to_event_ci_lower_1 >= 100_000:
        median_time_to_event_ci_lower_str_1 = r"n/a"
    else:
        median_time_to_event_ci_lower_str_1 = str(
            int(np.ceil(median_time_to_event_ci_lower_1))
        )
    if median_time_to_event_ci_upper_1 >= 100_000:
        median_time_to_event_ci_upper_str_1 = r"n/a"
    else:
        median_time_to_event_ci_upper_str_1 = str(
            int(np.ceil(median_time_to_event_ci_upper_1))
        )
    if median_time_to_event_2 >= 100_000:
        median_time_to_event_str_2 = r"n/a"
    else:
        median_time_to_event_str_2 = str(int(np.ceil(median_time_to_event_2)))
    if median_time_to_event_ci_lower_2 >= 200_000:
        median_time_to_event_ci_lower_str_2 = r"n/a"
    else:
        median_time_to_event_ci_lower_str_2 = str(
            int(np.ceil(median_time_to_event_ci_lower_2))
        )
    if median_time_to_event_ci_upper_2 >= 100_000:
        median_time_to_event_ci_upper_str_2 = r"n/a"
    else:
        median_time_to_event_ci_upper_str_2 = str(
            int(np.ceil(median_time_to_event_ci_upper_2))
        )

    p_logrank = lifelines.statistics.logrank_test(
        durations_A=durations_1,
        durations_B=durations_2,
        event_observed_A=observed_1,
        event_observed_B=observed_2,
    ).p_value
    p_logrank = np.round(p_logrank, 3)
    if p_logrank < 0.001:
        p_logrank_str = "< 0.001"
    else:
        p_logrank_str = str(p_logrank)

    ax.legend(
        title="Kaplan-Meier estimate",
        labels=[
            "Definition 1 (left)",
            "Definition 2 (right)",
        ],
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=BASE_PLOT_COLOR,
                linewidth=BASE_PLOT_LINEWIDTH,
            ),
            plt.Line2D(
                [0],
                [0],
                color=OVERLAY_PLOT_COLOR,
                linewidth=OVERLAY_PLOT_LINEWIDTH,
            ),
        ],
    )

    ax.set_xlim(xlim)
    ax.set_ylim((0, 1))

    ax.set_xlabel("Days after baseline")

    ax.set_title(
        "Event rate definition 1: "
        + str(np.round(event_rate_1 * 100, 2))
        + "%"
        + ", event rate definition 2: "
        + str(np.round(event_rate_2 * 100, 2))
        + "%"
        + "\nMedian time to first progression (95%CI) definition 1: "
        + median_time_to_event_str_1
        + " days ("
        + median_time_to_event_ci_lower_str_1
        + " - "
        + median_time_to_event_ci_upper_str_1
        + " days)"
        + "\nMedian time to first progression (95%CI) definition 2: "
        + median_time_to_event_str_2
        + " days ("
        + median_time_to_event_ci_lower_str_2
        + " - "
        + median_time_to_event_ci_upper_str_2
        + " days)"
        + "\np log-rank: "
        + p_logrank_str
    )

    sns.despine(bottom=True, left=True, right=True, top=True)

    st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    pass
