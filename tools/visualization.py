"""TBD"""

import numpy as np
import pandas as pd

from definitions import edssprogression

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

sns.set_theme(color_codes=True)
sns.set_style("whitegrid", {"grid.color": "gainsboro"})

# TODO: For 'end' option, stop drawing RAW/PIRA baselines


def annotate_plot_follow_up(
    follow_up_dataframe,
    relapse_timestamps=[],
    undefined_progression="re-baselining only",  # or "never", "all", "end"
    undefined_progression_wrt_raw_pira_baseline="greater only",  # "equal or greater", "any"
    return_first_event_only=False,
    merge_continuous_events=False,
    continuous_events_max_repetition_time=90,
    continuous_events_max_merge_distance=np.inf,
    opt_baseline_type="roving",
    opt_roving_reference_require_confirmation=True,
    opt_roving_reference_confirmation_time=0.5,  # amounts to next confirmed
    opt_roving_reference_confirmation_included_values="all",  # "last" or "all"
    opt_roving_reference_confirmation_time_right_side_max_tolerance=np.inf,
    opt_roving_reference_confirmation_time_left_side_max_tolerance=0,
    # PIRA/RAW options
    opt_raw_before_relapse_max_days=30,
    opt_raw_after_relapse_max_days=90,
    opt_pira_allow_relapses_between_event_and_confirmation=False,
    # Minimum increase options
    opt_max_score_that_requires_plus_1=5.0,
    opt_larger_increment_from_0=True,
    # Confirmation options
    opt_require_confirmation=False,
    opt_confirmation_time=-1,  # -1 for sustained over follow-up
    opt_confirmation_type="minimum",  # "minimum" or "monotonic"
    opt_confirmation_included_values="all",  # "last" or "all"
    opt_confirmation_sustained_minimal_distance=0,  # only if "sustained"
    opt_confirmation_time_right_side_max_tolerance=np.inf,  # not for "sustained"
    opt_confirmation_time_left_side_max_tolerance=0,  # not for "sustained"
    opt_confirmation_require_confirmation_for_last_visit=True,
    # Minimal distance options
    opt_minimal_distance_time=0,
    opt_minimal_distance_type="reference",  # "reference" or "previous"
    opt_minimal_distance_backtrack_decrease=True,  # go back to last low enough reference
    # Input specifications
    edss_score_column_name="edss_score",
    time_column_name="days_after_baseline",
    time_since_last_relapse_column_name="days_since_previous_relapse",
    time_to_next_relapse_column_name="days_to_next_relapse",
    # Output specifications
    is_general_rebaseline_flag_column_name="is_general_rebaseline",
    is_raw_pira_rebaseline_flag_column_name="is_raw_pira_rebaseline",
    is_post_relapse_rebaseline_flag_column_name="is_post_relapse_rebaseline",
    is_post_event_rebaseline_flag_column_name="is_post_event_rebaseline",
    used_as_general_reference_score_flag_column_name="edss_score_used_as_new_general_reference",
    used_as_raw_pira_reference_score_flag_column_name="edss_score_used_as_new_raw_pira_reference",
    is_progression_flag_column_name="is_progression",
    progression_type_column_name="progression_type",
    progression_score_column_name="progression_score",
    progression_event_id_column_name="progression_event_id",
    label_undefined_progression="Undefined",
    label_pira="PIRA",
    label_pira_confirmed_in_raw_window="PIRA confirmed in RAW window",
    label_raw="RAW",
    # Plot settings
    edss_color="black",
    relapse_color="deeppink",
    relapse_color_with_alpha="#FFC4E3",
    general_baseline_color="grey",
    raw_pira_baseline_color="deeppink",
    pira_color="#648FFF",
    label_pira_confirmed_in_raw_window_color="#785EF0",
    raw_color="#DC267F",
    undef_color="#FE6100",
    xlabel="Time",
    show_baselines=True,
    show_raw_window=True,
    show_progression=True,
    show_rebaselining=False,
    show_legend=True,
    move_legend_out=True,
    legend_loc="best",
    make_emf_safe=False,  # This replaces alpha with a solid lighter color.
    ax=None,
):
    # Setup ax
    if ax is None:
        ax = plt.gca()

    # Instantiate progression finder
    progression_finder = edssprogression.EDSSProgression(
        undefined_progression=undefined_progression,
        undefined_progression_wrt_raw_pira_baseline=undefined_progression_wrt_raw_pira_baseline,
        return_first_event_only=return_first_event_only,
        merge_continuous_events=merge_continuous_events,
        continuous_events_max_repetition_time=continuous_events_max_repetition_time,
        continuous_events_max_merge_distance=continuous_events_max_merge_distance,
        opt_baseline_type=opt_baseline_type,
        opt_roving_reference_require_confirmation=opt_roving_reference_require_confirmation,
        opt_roving_reference_confirmation_time=opt_roving_reference_confirmation_time,
        opt_roving_reference_confirmation_included_values=opt_roving_reference_confirmation_included_values,
        opt_roving_reference_confirmation_time_right_side_max_tolerance=opt_roving_reference_confirmation_time_right_side_max_tolerance,
        opt_roving_reference_confirmation_time_left_side_max_tolerance=opt_roving_reference_confirmation_time_left_side_max_tolerance,
        # PIRA/RAW options
        opt_raw_before_relapse_max_days=opt_raw_before_relapse_max_days,
        opt_raw_after_relapse_max_days=opt_raw_after_relapse_max_days,
        opt_pira_allow_relapses_between_event_and_confirmation=opt_pira_allow_relapses_between_event_and_confirmation,
        # Minimum increase options
        opt_max_score_that_requires_plus_1=opt_max_score_that_requires_plus_1,
        opt_larger_increment_from_0=opt_larger_increment_from_0,
        # Confirmation options
        opt_require_confirmation=opt_require_confirmation,
        opt_confirmation_time=opt_confirmation_time,
        opt_confirmation_type=opt_confirmation_type,
        opt_confirmation_included_values=opt_confirmation_included_values,
        opt_confirmation_sustained_minimal_distance=opt_confirmation_sustained_minimal_distance,
        opt_confirmation_time_right_side_max_tolerance=opt_confirmation_time_right_side_max_tolerance,
        opt_confirmation_time_left_side_max_tolerance=opt_confirmation_time_left_side_max_tolerance,
        opt_confirmation_require_confirmation_for_last_visit=opt_confirmation_require_confirmation_for_last_visit,
        # Minimal distance options
        opt_minimal_distance_time=opt_minimal_distance_time,
        opt_minimal_distance_type=opt_minimal_distance_type,
        opt_minimal_distance_backtrack_decrease=opt_minimal_distance_backtrack_decrease,
        # Input specifications
        edss_score_column_name=edss_score_column_name,
        time_column_name=time_column_name,
        time_since_last_relapse_column_name=time_since_last_relapse_column_name,
        time_to_next_relapse_column_name=time_to_next_relapse_column_name,
        # Output specifications
        is_general_rebaseline_flag_column_name=is_general_rebaseline_flag_column_name,
        is_raw_pira_rebaseline_flag_column_name=is_raw_pira_rebaseline_flag_column_name,
        is_post_relapse_rebaseline_flag_column_name=is_post_relapse_rebaseline_flag_column_name,
        is_post_event_rebaseline_flag_column_name=is_post_event_rebaseline_flag_column_name,
        used_as_general_reference_score_flag_column_name=used_as_general_reference_score_flag_column_name,
        used_as_raw_pira_reference_score_flag_column_name=used_as_raw_pira_reference_score_flag_column_name,
        is_progression_flag_column_name=is_progression_flag_column_name,
        progression_type_column_name=progression_type_column_name,
        progression_score_column_name=progression_score_column_name,
        progression_event_id_column_name=progression_event_id_column_name,
        label_undefined_progression=label_undefined_progression,
        label_pira=label_pira,
        label_pira_confirmed_in_raw_window=label_pira_confirmed_in_raw_window,
        label_raw=label_raw,
    )

    # Annotate baselines and progression
    annotated_df = progression_finder.add_progression_events_to_follow_up(
        follow_up_dataframe,
        relapse_timestamps=relapse_timestamps,
    )

    # Min and max score, used for styling later
    min_edss = annotated_df[edss_score_column_name].min() - 0.5
    max_edss = annotated_df[edss_score_column_name].max() + 0.5

    # Get relapse timestamps
    if len(relapse_timestamps) > 0:
        # Determine height of axvspan
        height_of_plot = max_edss - min_edss + 0.5
        values_range = max_edss - min_edss
        fraction_not_covered = (1 - values_range / height_of_plot) / 2
        for relapse_timestamp in relapse_timestamps:
            # Marks a relapse
            ax.vlines(
                x=relapse_timestamp,
                ymin=min_edss,
                ymax=max_edss,
                color=relapse_color,
                linewidth=1.5,
                zorder=2,
            )
            # Highlights the period not available for re-baselining
            # or PIRA. NOTE: The custom color is used because alpha
            # does not seem to work when exporting to .svg and then
            # converting to .emf.
            if show_raw_window:
                if make_emf_safe:
                    ax.axvspan(
                        relapse_timestamp - opt_raw_before_relapse_max_days,
                        relapse_timestamp + opt_raw_after_relapse_max_days,
                        ymin=fraction_not_covered,
                        ymax=1 - fraction_not_covered,
                        linewidth=0,
                        color=relapse_color_with_alpha,
                        zorder=-1,
                    )
                else:
                    ax.axvspan(
                        relapse_timestamp - opt_raw_before_relapse_max_days,
                        relapse_timestamp + opt_raw_after_relapse_max_days,
                        ymin=fraction_not_covered,
                        ymax=1 - fraction_not_covered,
                        linewidth=0,
                        alpha=0.25,
                        color=relapse_color,
                        zorder=2,
                    )

    # EDSS
    sns.lineplot(
        annotated_df,
        x=time_column_name,
        y=edss_score_column_name,
        color=edss_color,
        marker="o",
        linewidth=2.5,
        zorder=3,
        ax=ax,
    )

    if show_baselines:
        if len(relapse_timestamps) > 0:
            rbl = annotated_df[
                annotated_df[is_post_relapse_rebaseline_flag_column_name]
            ]
            if len(rbl) > 0:
                sns.lineplot(
                    annotated_df[
                        annotated_df[is_post_relapse_rebaseline_flag_column_name]
                    ],
                    x=time_column_name,
                    y=edss_score_column_name,
                    color=relapse_color,
                    marker="o",
                    linewidth=0,
                    zorder=6,
                    ax=ax,
                )

        # Baselines
        baseline_specs_dict_list = [
            {
                "flag": is_general_rebaseline_flag_column_name,
                "score": used_as_general_reference_score_flag_column_name,
                "color": general_baseline_color,
                "marker": "D",
                "linewidth": 1.5,
                "zorder": 4,
            },
        ]
        if len(relapse_timestamps) > 0:
            baseline_specs_dict_list = baseline_specs_dict_list + [
                {
                    "flag": is_raw_pira_rebaseline_flag_column_name,
                    "score": used_as_raw_pira_reference_score_flag_column_name,
                    "color": raw_pira_baseline_color,
                    "marker": "*",
                    "linewidth": 1,
                    "zorder": 5,
                }
            ]

        for baseline_specs in baseline_specs_dict_list:
            # Baseline - markers for each assessment where re-baselining happens
            rebaselines = (
                annotated_df[annotated_df[baseline_specs["flag"]]].reset_index().copy()
            )
            if show_rebaselining:
                sns.lineplot(
                    data=rebaselines,
                    x=time_column_name,
                    y=edss_score_column_name,
                    color=baseline_specs["color"],
                    marker=baseline_specs["marker"],
                    linewidth=0,
                    markersize=8,
                    zorder=baseline_specs["zorder"],
                    ax=ax,
                )
            if len(rebaselines) == 0:
                # First
                ax.hlines(
                    xmin=annotated_df[time_column_name].min(),
                    xmax=annotated_df[time_column_name].max(),
                    y=annotated_df.iloc[0][edss_score_column_name],
                    color=baseline_specs["color"],
                    zorder=1,
                    linewidth=baseline_specs["linewidth"],
                )
            elif len(rebaselines) > 0:
                # Baseline - add lines that indicate the basline
                for i, row in rebaselines.iloc[:-1].iterrows():
                    ax.hlines(
                        xmin=row[time_column_name],
                        xmax=rebaselines.loc[i + 1][time_column_name],
                        y=row[baseline_specs["score"]],
                        color=baseline_specs["color"],
                        zorder=1,
                        linewidth=baseline_specs["linewidth"],
                    )
                # First
                ax.hlines(
                    xmin=annotated_df[time_column_name].min(),
                    xmax=rebaselines.iloc[0][time_column_name],
                    y=annotated_df.iloc[0][edss_score_column_name],
                    color=baseline_specs["color"],
                    zorder=1,
                    linewidth=baseline_specs["linewidth"],
                )
                # Last
                ax.hlines(
                    xmin=rebaselines.iloc[-1][time_column_name],
                    xmax=annotated_df[time_column_name].max(),
                    y=rebaselines.iloc[-1][baseline_specs["score"]],
                    color=baseline_specs["color"],
                    zorder=1,
                    linewidth=baseline_specs["linewidth"],
                )
                # First vertical
                ax.vlines(
                    x=rebaselines.iloc[0][time_column_name],
                    ymin=annotated_df.iloc[0][edss_score_column_name],
                    ymax=rebaselines.iloc[0][baseline_specs["score"]],
                    color=baseline_specs["color"],
                    zorder=1,
                    linewidth=baseline_specs["linewidth"],
                )
                for i, row in rebaselines.iloc[1:].iterrows():
                    ax.vlines(
                        x=row[time_column_name],
                        ymin=min(
                            rebaselines.loc[i - 1][baseline_specs["score"]],
                            row[baseline_specs["score"]],
                        ),
                        ymax=max(
                            rebaselines.loc[i - 1][baseline_specs["score"]],
                            row[baseline_specs["score"]],
                        ),
                        color=baseline_specs["color"],
                        zorder=1,
                        linewidth=baseline_specs["linewidth"],
                    )

    # Progression
    if show_progression:
        sns.lineplot(
            data=annotated_df[
                (annotated_df[is_progression_flag_column_name])
                & (
                    annotated_df[progression_type_column_name]
                    == label_undefined_progression
                )
            ],
            x=time_column_name,
            y=progression_score_column_name,
            color=undef_color,
            linewidth=0,
            markersize=18,
            marker="*",
            zorder=7,
            ax=ax,
        )
        undef_progression_ids = annotated_df[
            (annotated_df[is_progression_flag_column_name])
            & (
                annotated_df[progression_type_column_name]
                == label_undefined_progression
            )
        ][progression_event_id_column_name]
        sns.lineplot(
            data=annotated_df[
                annotated_df[progression_event_id_column_name].isin(
                    undef_progression_ids
                )
            ],
            x=time_column_name,
            y=edss_score_column_name,
            color=undef_color,
            linewidth=0,
            marker="o",
            zorder=7,
            ax=ax,
        )

        # Progression
        sns.lineplot(
            data=annotated_df[
                (annotated_df[is_progression_flag_column_name])
                & (annotated_df[progression_type_column_name] == label_raw)
            ],
            x=time_column_name,
            y=progression_score_column_name,
            color=raw_color,
            linewidth=0,
            markersize=18,
            marker="*",
            zorder=7,
            ax=ax,
        )
        raw_progression_ids = annotated_df[
            (annotated_df[is_progression_flag_column_name])
            & (annotated_df[progression_type_column_name] == label_raw)
        ][progression_event_id_column_name]
        sns.lineplot(
            data=annotated_df[
                annotated_df[progression_event_id_column_name].isin(raw_progression_ids)
            ],
            x=time_column_name,
            y=edss_score_column_name,
            color=raw_color,
            linewidth=0,
            marker="o",
            zorder=7,
            ax=ax,
        )

        # Progression
        sns.lineplot(
            data=annotated_df[
                (annotated_df[is_progression_flag_column_name])
                & (annotated_df[progression_type_column_name] == label_pira)
            ],
            x=time_column_name,
            y=progression_score_column_name,
            color=pira_color,
            linewidth=0,
            markersize=18,
            marker="*",
            zorder=7,
            ax=ax,
        )
        pira_progression_ids = annotated_df[
            (annotated_df[is_progression_flag_column_name])
            & (annotated_df[progression_type_column_name] == label_pira)
        ][progression_event_id_column_name]
        sns.lineplot(
            data=annotated_df[
                annotated_df[progression_event_id_column_name].isin(
                    pira_progression_ids
                )
            ],
            x=time_column_name,
            y=edss_score_column_name,
            color=pira_color,
            linewidth=0,
            marker="o",
            zorder=7,
            ax=ax,
        )

        # Progression
        sns.lineplot(
            data=annotated_df[
                (annotated_df[is_progression_flag_column_name])
                & (
                    annotated_df[progression_type_column_name]
                    == label_pira_confirmed_in_raw_window
                )
            ],
            x=time_column_name,
            y=progression_score_column_name,
            color=label_pira_confirmed_in_raw_window_color,
            linewidth=0,
            markersize=18,
            marker="*",
            zorder=7,
            ax=ax,
        )
        pira_raw_progression_ids = annotated_df[
            (annotated_df[is_progression_flag_column_name])
            & (
                annotated_df[progression_type_column_name]
                == label_pira_confirmed_in_raw_window
            )
        ][progression_event_id_column_name]
        sns.lineplot(
            data=annotated_df[
                annotated_df[progression_event_id_column_name].isin(
                    pira_raw_progression_ids
                )
            ],
            x=time_column_name,
            y=edss_score_column_name,
            color=label_pira_confirmed_in_raw_window_color,
            linewidth=0,
            marker="o",
            zorder=7,
            ax=ax,
        )

    # Legend
    if show_legend:
        legend_handles = [
            plt.Line2D([], [], color=edss_color, marker="o", linewidth=0),
        ]
        legend_labels = [
            "EDSS assessment",
        ]
        if len(relapse_timestamps) > 0:
            legend_handles = legend_handles + [
                plt.Line2D(
                    [], [], color=relapse_color, marker="|", markersize=12, linewidth=0
                ),
            ]
            legend_labels = legend_labels + [
                "Relapse",
            ]

        if show_baselines:
            legend_handles = legend_handles + [
                plt.Line2D(
                    [], [], color=general_baseline_color, marker=None, linewidth=1
                ),
            ]
            legend_labels = legend_labels + [
                "Relapse-independent reference",
            ]

            if len(relapse_timestamps) > 0:
                legend_handles = legend_handles + [
                    plt.Line2D(
                        [], [], color=raw_pira_baseline_color, marker=None, linewidth=1
                    ),
                    plt.Line2D(
                        [], [], color=raw_pira_baseline_color, marker="o", linewidth=0
                    ),
                ]
                legend_labels = legend_labels + [
                    "Reference for RAW/PIRA",
                    "Post-relapse re-baselining",
                ]

        if show_raw_window and (len(relapse_timestamps) > 0):
            if make_emf_safe:
                legend_handles = legend_handles + [
                    Patch(color=relapse_color_with_alpha, linewidth=0),
                ]
            else:
                legend_handles = legend_handles + [
                    Patch(color=relapse_color, alpha=0.25, linewidth=0),
                ]
            legend_labels = legend_labels + [
                "Pre-/post-relapse RAW window",
            ]

        if show_progression:
            legend_handles = legend_handles + [
                plt.Line2D([], [], color=pira_color, marker="o", linewidth=0),
                plt.Line2D(
                    [], [], color=pira_color, marker="*", markersize=12, linewidth=0
                ),
            ]
            legend_labels = legend_labels + [
                "PIRA event assessment",
                "PIRA event score",
            ]
            if len(relapse_timestamps) > 0:
                legend_handles = legend_handles + [
                    plt.Line2D(
                        [],
                        [],
                        color=label_pira_confirmed_in_raw_window_color,
                        marker="o",
                        linewidth=0,
                    ),
                    plt.Line2D(
                        [],
                        [],
                        color=label_pira_confirmed_in_raw_window_color,
                        marker="*",
                        markersize=12,
                        linewidth=0,
                    ),
                    plt.Line2D([], [], color=raw_color, marker="o", linewidth=0),
                    plt.Line2D(
                        [], [], color=raw_color, marker="*", markersize=12, linewidth=0
                    ),
                    plt.Line2D([], [], color=undef_color, marker="o", linewidth=0),
                    plt.Line2D(
                        [],
                        [],
                        color=undef_color,
                        marker="*",
                        markersize=12,
                        linewidth=0,
                    ),
                ]
                legend_labels = legend_labels + [
                    "PIRA with relapse during confirmation event assessment",
                    "PIRA with relapse during confirmation event score",
                    "RAW event assessment",
                    "RAW event score",
                    "Undefined worsening event assessment",
                    "Undefined worsening event score",
                ]

        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc=legend_loc,
        )
        if move_legend_out:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    else:
        ax.legend([], [], frameon=False)

    ax.set_xticks(annotated_df[time_column_name])
    ax.set_yticks(
        [min_edss + 0.5 * i for i in range(int(np.ceil((max_edss - min_edss) * 2)) + 1)]
    )
    ax.set_ylim((min_edss - 0.25, max_edss + 0.25))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("EDSS")

    return ax


if __name__ == "__main__":
    pass
