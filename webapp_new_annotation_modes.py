"""Streamlit webapp source file.

This is the source code for the Steramlit webapp deployed on
https://multiple-sclerosis-disability-progression.streamlit.app/

"""

from datetime import date, datetime, timezone
from io import BytesIO

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import xlsxwriter
from definitions import edssannotation
from matplotlib import figure
from tools import evaluation_new_annotation_modes as evaluation
from tools import visualization_new_annotation_modes as visualization
from webapp_toolbox import cached_functions
from webapp_toolbox import frontend_new_annotation_modes as frontend

matplotlib.use("Agg")
sns.set_theme(color_codes=True)
sns.set_style("whitegrid", {"grid.color": "gainsboro"})


# Wide layout
st.set_page_config(layout="wide")


# Cached clock to help keeping track of cache misses
@st.cache_data()
def show_clock_last_cache_refresh():
    local_timezone = datetime.now(timezone.utc).astimezone().tzinfo
    return (
        datetime.now(tz=local_timezone).strftime("%d.%m.%Y, %H:%M:%S")
        + " "
        + str(local_timezone)
    )


# Cached EDSS disability accrual annotation instance
@st.cache_data(ttl=60)
def instantiate_annotator(
    annotation_mode="symmetric",
    undefined_events_annotation_mode="all",
    return_first_event_only=False,
    merge_continuous_events=False,
    continuous_events_max_repetition_time=30,
    continuous_events_max_merge_distance=np.inf,
    opt_baseline_type="fixed",
    opt_roving_reference_require_confirmation=True,
    opt_roving_reference_confirmation_time=0.5,
    opt_roving_reference_confirmation_included_values="all",
    opt_roving_reference_confirmation_time_left_side_max_tolerance=0,
    opt_roving_reference_confirmation_time_right_side_max_tolerance=np.inf,
    opt_raw_before_relapse_max_time=30,
    opt_raw_after_relapse_max_time=90,
    opt_pira_allow_relapses_between_event_and_confirmation=False,
    opt_max_score_that_requires_plus_1=5.0,
    opt_larger_increment_from_0=True,
    opt_require_confirmation=True,
    opt_confirmation_time=6 * 30,
    opt_confirmation_type="minimum",
    opt_confirmation_included_values="all",
    opt_confirmation_sustained_minimal_distance=0,
    opt_confirmation_time_left_side_max_tolerance=0,
    opt_confirmation_time_right_side_max_tolerance=np.inf,
    opt_confirmation_require_confirmation_for_last_visit=True,
    opt_minimal_distance_time=0,
    opt_minimal_distance_type="reference",
    opt_minimal_distance_backtrack_decrease=True,
):
    return edssannotation.EDSSAnnotation(
        annotation_mode=annotation_mode,
        undefined_events_annotation_mode=undefined_events_annotation_mode,
        return_first_event_only=return_first_event_only,
        merge_continuous_events=merge_continuous_events,
        continuous_events_max_repetition_time=continuous_events_max_repetition_time,
        continuous_events_max_merge_distance=continuous_events_max_merge_distance,
        opt_baseline_type=opt_baseline_type,
        opt_roving_reference_require_confirmation=opt_roving_reference_require_confirmation,
        opt_roving_reference_confirmation_time=opt_roving_reference_confirmation_time,
        opt_roving_reference_confirmation_included_values=opt_roving_reference_confirmation_included_values,
        opt_roving_reference_confirmation_time_left_side_max_tolerance=opt_roving_reference_confirmation_time_left_side_max_tolerance,
        opt_roving_reference_confirmation_time_right_side_max_tolerance=opt_roving_reference_confirmation_time_right_side_max_tolerance,
        opt_raw_before_relapse_max_time=opt_raw_before_relapse_max_time,
        opt_raw_after_relapse_max_time=opt_raw_after_relapse_max_time,
        opt_pira_allow_relapses_between_event_and_confirmation=opt_pira_allow_relapses_between_event_and_confirmation,
        opt_max_score_that_requires_plus_1=opt_max_score_that_requires_plus_1,
        opt_larger_increment_from_0=opt_larger_increment_from_0,
        opt_require_confirmation=opt_require_confirmation,
        opt_confirmation_time=opt_confirmation_time,
        opt_confirmation_type=opt_confirmation_type,
        opt_confirmation_included_values=opt_confirmation_included_values,
        opt_confirmation_sustained_minimal_distance=opt_confirmation_sustained_minimal_distance,
        opt_confirmation_time_left_side_max_tolerance=opt_confirmation_time_left_side_max_tolerance,
        opt_confirmation_time_right_side_max_tolerance=opt_confirmation_time_right_side_max_tolerance,
        opt_confirmation_require_confirmation_for_last_visit=opt_confirmation_require_confirmation_for_last_visit,
        opt_minimal_distance_time=opt_minimal_distance_time,
        opt_minimal_distance_type=opt_minimal_distance_type,
        opt_minimal_distance_backtrack_decrease=opt_minimal_distance_backtrack_decrease,
    )


# Cached evaluation instance
@st.cache_data(ttl=60)
def cached_evaluation_instance():
    return evaluation.EDSSAnnotationEvaluation()


# Cached annotated dataframe
@st.cache_data(ttl=60)
def cached_annotated_df(annotator_instance, follow_up_dataframe, relapse_timestamps):
    return annotator_instance.add_event_annotation_to_follow_up(
        follow_up_dataframe=follow_up_dataframe, relapse_timestamps=relapse_timestamps
    )


# Cached uploaded file
@st.cache_data(ttl=60)
def load_excel_table(uploaded_file):
    return cached_functions.load_excel_table(uploaded_file=uploaded_file)


# Example follow-up and relapse timestamps
example_follow_up_df = pd.DataFrame(
    [
        {"days_after_baseline": 0, "edss_score": 1.0},
        {"days_after_baseline": 60, "edss_score": 2.5},
        {"days_after_baseline": 90, "edss_score": 1.5},
        {"days_after_baseline": 150, "edss_score": 2.0},
        {"days_after_baseline": 220, "edss_score": 3.5},
        {"days_after_baseline": 250, "edss_score": 2.0},
        {"days_after_baseline": 310, "edss_score": 2.0},
        {"days_after_baseline": 350, "edss_score": 4.0},
        {"days_after_baseline": 380, "edss_score": 3.5},
        {"days_after_baseline": 420, "edss_score": 3.5},
        {"days_after_baseline": 480, "edss_score": 2.5},
        {"days_after_baseline": 540, "edss_score": 2.5},
        {"days_after_baseline": 600, "edss_score": 3.5},
        {"days_after_baseline": 660, "edss_score": 4.0},
        {"days_after_baseline": 720, "edss_score": 4.5},
        {"days_after_baseline": 780, "edss_score": 5.0},
        {"days_after_baseline": 840, "edss_score": 5.0},
        {"days_after_baseline": 900, "edss_score": 5.0},
    ]
)
example_relapse_timestamps = [40, 200, 330, 770]

if __name__ == "__main__":
    # Setup general app layout
    st.title(
        "Disability Accrual and Improvement in Multiple Sclerosis from EDSS Scores"
    )
    st.markdown(
        "A little playground to explore the myriads of possible definitions of *EDSS disability accrual and improvement* "
        + "as implemented in [multiple-sclerosis-disability-progression](https://github.com/drstrupf/multiple-sclerosis-disability-progression)."
    )

    # More detailed author information
    with st.expander("Author information and contact", expanded=False):
        st.markdown("### Authors")
        st.markdown(
            r"**Gabriel Bsteh**<sup>1, 2</sup>, **Stefanie Marti**<sup>3</sup>, **Robert Hoepner**<sup>3, 4</sup>",
            unsafe_allow_html=True,
        )
        st.markdown(
            r"""<sup>1</sup>Department of Neurology, Medical University of Vienna, Vienna, Austria  
        <sup>2</sup>Comprehensive Center for Clinical Neurosciences and Mental Health, Medical University of Vienna, Vienna, Austria  
        <sup>3</sup>Department of Neurology, Inselspital, Bern University Hospital and University of Bern, Switzerland  
        <sup>4</sup>Insel Data Science Center, Inselspital, Bern University Hospital and University of Bern, Switzerland""",
            unsafe_allow_html=True,
        )
        st.markdown("### Contact information")
        st.markdown(
            r"""Found a **bug**? Do you have a **feature request**? We would appreciate your feedback!
            Please **open an issue** on our [GitHub project page](https://github.com/drstrupf/multiple-sclerosis-disability-progression) or
            **contact the authors** ([Gabriel Bsteh](https://www.meduniwien.ac.at/web/forschung/researcher-profiles/researcher-profiles/detail/?res=gabriel_bsteh&cHash=0896fd3f091c51c7c5c37b55b83d8def),
            [Robert Hoepner](http://www.neurologie.insel.ch/de/ueber-uns/teams/details/person/detail/robert-hoepner))."""
        )

    # Cite the papers
    with st.expander(
        "Related publications",
        expanded=False,
    ):
        # RMS
        st.markdown(
            "### Dissecting definitions of disability accrual in relapsing multiple sclerosis—Have we reached standardization yet?"
        )
        st.markdown("##### Abstract")
        st.markdown(
            "**Background**: Distinguishing relapse-associated worsening (RAW) and progression independent of relapse activity (PIRA) has reshaped understanding  "
            + "of disability accumulation in relapsing multiple sclerosis (RMS). The influence of differing definitions of disability accrual on event rates and RAW/PIRA  "
            + "proportions remains uncertain.  "
            + "\n\n**Methods**: This observational cohort study used Austrian MS Treatment Registry data (2010–2024). A custom algorithm evaluated 1440 definitional  "
            + "variants of disability accrual with varying confirmation duration, baseline modeling, and RAW/PIRA classification, including recently proposed  "
            + "“standardized” criteria.  "
            + "\n\n**Results**: We included 3273 RMS patients (mean age 37.5 years; 67.8% female) with ⩾24 months follow-up, ⩾3 Expanded Disability Status Scale (EDSS)  "
            + "scores, and ⩾1 EDSS score per year, contributing 3525 follow-up periods. Depending on definition, disability accrual varied between 15.7% and 41.6% of  "
            + "follow-ups. PIRA accounted for 56.1%–86.3% of events across definitions, while up to 8.4% were ambiguously classified, mainly due to post-relapse  "
            + "re-baselining or relapses during the confirmation period. Even under “standardized” criteria, 144 definitional combinations remained, with event rates  "
            + "ranging from 19.1% to 21.7% and PIRA contribution varying widely from 59.8% to 85.8%."
            + "\n\n**Conclusions**: PIRA predominantly drives disability accrual, yet definitional variation substantially influenced event rates and RAW/PIRA proportions.  "
            + "Transparent reporting and further optimization of definitions are critical for improving comparability, interpretation, and clinical relevance in MS  "
            + "research and care.  "
        )
        st.markdown("##### Reference")
        st.markdown(
            "Bsteh G, Marti S, Hammer H, Krajnc N, Guger M,  Di Pauli F, Kraus J, Enzinger C, Chan A, Berger T, Hegen H, Hoepner R.  "
            + "\n**Dissecting definitions of disability accrual in relapsing multiple sclerosis—Have we reached standardization yet?.**  "
            + "\n*Mult Scler*. 2026 Feb;32(2):179-191. doi: 10.1177/13524585251396283. Epub 2025 Dec 6. [PMID: 41351456](https://pubmed.ncbi.nlm.nih.gov/41351456/)."
        )
        # PPMS
        st.markdown(
            "### Disability progression is a question of definition - A methodological reappraisal by example of primary progressive multiple sclerosis"
        )
        st.markdown("##### Abstract")
        st.markdown(
            "**Background**: Different definitions of disability progression by Expanded Disability Status Scale (EDSS) may influence frequency and/or time to event.  "
            + "\n\n**Methods**: In this multicenter cohort study, we included PPMS patients with follow-up ≥24 months and ≥3 available EDSS scores overall (≥1 per year).  "
            + "We applied 672 definitions of disability progression including different minimal EDSS increase, required confirmation and fixed/roving-baseline score.  "
            + "\n\n**Results**: We analyzed follow-up periods from 131 PPMS patients (median age at baseline 53.0 years [45.0 - 63.0], 51.9 % female, median follow-up 3.9  "
            + "years [2.6 - 6.0], median baseline EDSS 4.0 [2.5 - 6.0]). The most sensitive definition of a progression event was an unconfirmed increase of ≥0.5 points   "
            + "with a roving baseline (81.8 % event rate). The least sensitive definition was an increase of ≥1.0 points with a fixed baseline, minimal distance to reference  "
            + "48 weeks, and confirmed at ≥48 weeks (28.4 % event rate). Comparing roving vs. fixed baseline over all cutoffs and confirmation definitions, average time  "
            + "to progression was 227 days shorter applying the roving baseline (1405 days [550 - 2653] vs. 1632 days [760 - 2653]).  "
            + "\n\n**Conclusions**: Different definitions of disability progression result in significantly differing rates of disability progression, which may influence   "
            + "study results and create confusion in clinical practice."
        )
        st.markdown("##### Reference")
        st.markdown(
            "Bsteh G, Marti S, Krajnc N, Traxler G, Salmen A, Hammer H, Leutmezer F, Rommer P, Di Pauli F, Chan A, Berger T, Hegen H, Hoepner R.  "
            + "\n**Disability progression is a question of definition - A methodological reappraisal by example of primary progressive multiple sclerosis.**  "
            + "\n*Mult Scler Relat Disord*. 2025 Jan;93:106215. doi: 10.1016/j.msard.2024.106215. Epub 2024 Dec 6. [PMID: 39662164](https://pubmed.ncbi.nlm.nih.gov/39662164/)."
        )

    st.write("Last cache refresh: " + show_clock_last_cache_refresh())

    # Expander with the example for playing around
    st.write("## Explore accrual, improvement, PIRA, and RAW definition options")
    st.markdown(
        "This section illustrates how **relapse-related options** as well as **general accrual/improvement options** affect the number of "
        + " events, the times to event, and the event types."
    )

    with st.expander(
        "Plot follow-up and annotate events for example data",
        expanded=True,
    ):
        data_edit_column, option_selection_column, plot_column = st.columns(
            [15, 30, 55]
        )

        with data_edit_column:
            st.write("Edit example dataframe")

            edited_example_follow_up_df = frontend.example_input_dataframe_editor(
                follow_up_dataframe=example_follow_up_df,
                element_base_key="relapse_example_follow_up_editor_key",
            )

        with option_selection_column:
            st.write("Select definition options")
            options_example = frontend.dynamic_progression_option_input_element(
                element_base_key="plot_playground_options_example",
                default_annotation_mode="symmetric",
                default_undefined_events_annotation_mode="all",
                default_baseline="roving",
                default_confirmation_requirement=True,
                default_confirmation_duration=30,
                display_rms_options=True,
                display_allow_relapses_in_pira_conf=False,
            )

        with plot_column:
            st.write(
                "Add, remove, or change relapses and display the annotated follow-up"
            )
            relapse_timestamps = st.multiselect(
                label="Relapses (days after baseline)",
                options=[i for i in range(1000)],
                default=example_relapse_timestamps,
                key="add_relapses_widget",
                help=None,
                on_change=None,
                max_selections=None,
                placeholder="Add relapses!",
                disabled=False,
                label_visibility="visible",
            )
            # Instantiate an annotator
            annotator_instance = instantiate_annotator(
                # Options
                annotation_mode=options_example["annotation_mode"],
                undefined_events_annotation_mode=options_example[
                    "undefined_events_annotation_mode"
                ],
                opt_raw_before_relapse_max_time=options_example[
                    "opt_raw_before_relapse_max_time"
                ],
                opt_raw_after_relapse_max_time=options_example[
                    "opt_raw_after_relapse_max_time"
                ],
                opt_pira_allow_relapses_between_event_and_confirmation=options_example.get(
                    "opt_pira_allow_relapses_between_event_and_confirmation", False
                ),
                opt_baseline_type=options_example["opt_baseline_type"],
                opt_roving_reference_require_confirmation=options_example[
                    "opt_roving_reference_require_confirmation"
                ],
                opt_roving_reference_confirmation_time=options_example[
                    "opt_roving_reference_confirmation_time"
                ],
                opt_max_score_that_requires_plus_1=options_example[
                    "opt_increase_threshold"
                ],
                opt_larger_increment_from_0=options_example[
                    "opt_larger_minimal_increase_from_0"
                ],
                opt_minimal_distance_time=options_example["opt_minimal_distance_time"],
                opt_minimal_distance_type=options_example["opt_minimal_distance_type"],
                opt_minimal_distance_backtrack_decrease=options_example[
                    "opt_minimal_distance_backtrack_decrease"
                ],
                opt_require_confirmation=options_example["opt_require_confirmation"],
                opt_confirmation_time=options_example["opt_confirmation_time"],
                opt_confirmation_type=options_example["opt_confirmation_type"],
                opt_confirmation_included_values=options_example[
                    "opt_confirmation_included_values"
                ],
                opt_confirmation_sustained_minimal_distance=options_example[
                    "opt_confirmation_sustained_minimal_distance"
                ],
            )

            # Annotate the dataframe
            annotated_df = cached_annotated_df(
                annotator_instance=annotator_instance,
                follow_up_dataframe=edited_example_follow_up_df,
                relapse_timestamps=relapse_timestamps,
            )

            # Plot it
            fig = figure.Figure(figsize=(16, 6))
            ax = fig.subplots(1)
            visualization.plot_annotated_follow_up(
                annotated_df,
                annotation_mode=options_example["annotation_mode"],
                opt_raw_before_relapse_max_time=options_example[
                    "opt_raw_before_relapse_max_time"
                ],
                opt_raw_after_relapse_max_time=options_example[
                    "opt_raw_after_relapse_max_time"
                ],
                xlabel="Days after baseline",
                ax=ax,
            )
            fig.tight_layout()
            sns.despine(bottom=True, left=True, right=True, top=True, ax=ax)
            st.pyplot(fig, clear_figure=True)

            # Display overall stats
            Eval = cached_evaluation_instance()
            cohort_stats_overall_df = Eval.get_cohort_stats(
                stats_by_follow_up=Eval.get_follow_up_stats(
                    annotated_follow_ups=annotated_df,
                    get_stats_by_type=False,
                    id_columns=None,
                ),
                get_stats_by_type=False,
                follow_up_id_column=None,
                groupby_columns=None,
            )
            cohort_stats_overall_df_display = cohort_stats_overall_df[
                [
                    "total_events",
                    "total_accrual_events",
                    "total_improvement_events",
                    "total_event_score_delta",
                    "total_accrual_event_score_delta",
                    "total_improvement_event_score_delta",
                    "contribution_of_accrual_to_total_events",
                    "contribution_of_improvement_to_total_events",
                ]
            ].rename(
                columns={
                    "total_events": "Total events",
                    "total_accrual_events": "Accrual events",
                    "total_improvement_events": "Improvement events",
                    "total_event_score_delta": "EDSS delta",
                    "total_accrual_event_score_delta": "Accrual EDSS delta",
                    "total_improvement_event_score_delta": "Improvement EDSS delta",
                    "contribution_of_accrual_to_total_events": "Contribution of accrual to total events",
                    "contribution_of_improvement_to_total_events": "Contribution of improvement to total events",
                }
            )
            st.write("Overall results, scroll to the right for more columns")
            st.dataframe(cohort_stats_overall_df_display)

            # Display stats by event type
            follow_up_stats_df = Eval.get_follow_up_stats(
                annotated_follow_ups=annotated_df,
                get_stats_by_type=True,
                id_columns=None,
            )
            cohort_stats_df = Eval.get_cohort_stats(
                stats_by_follow_up=follow_up_stats_df,
                get_stats_by_type=True,
                follow_up_id_column=None,
                groupby_columns=None,
            )
            cohort_stats_df_display = cohort_stats_df[
                [
                    "event_type",
                    "total_events",
                    "total_event_score_delta",
                    "contribution_to_total_events",
                    "contribution_to_total_accrual_events",
                    "contribution_to_total_accrual_delta",
                ]
            ].rename(
                columns={
                    "event_type": "Event type",
                    "total_events": "Total events",
                    "total_event_score_delta": "Total EDSS delta",
                    "contribution_to_total_events": "Contribution to total events",
                    "contribution_to_total_accrual_events": "Contribution to accrual events",
                    "contribution_to_total_accrual_delta": "Contribution to accrual delta",
                }
            )
            st.write("Results by type, scroll to the right for more columns")
            st.dataframe(cohort_stats_df_display)

    # Expander for upload, annotation, and visualization of a single follow-up
    st.write("## Upload, annotate, and visualize your own example data")
    st.markdown(
        "**Experimental**: Upload your own example data and check the annotation results."
    )
    with st.expander(
        "Plot follow-up and annotate events for uploaded example data",
        expanded=False,
    ):
        st.write(
            "**Under development**. For now, it only works for a **single follow-up** and dataframes with columns ``edss_score`` and"
            + " ``days_after_baseline``, where days after baseline must be integers. Relapses can be provided by a second .xlsx file"
            + " with relapse timestamps as integers and column name ``days_after_baseline``. You can download the example data from"
            + " the playground section above in .xlsx format as reference/template (one sheet with the EDSS follow-ups, one with the"
            + " relapse timestamps)."
        )

        st.warning(
            "**Warning**: No sanity check upon upload; the app will crash if data are not well formatted!"
            " Removing the uploaded files will fix it."
        )

        st.write("**Download example data**")
        dl_file_buffer = BytesIO()
        with pd.ExcelWriter(dl_file_buffer, engine="xlsxwriter") as writer:
            edited_example_follow_up_df.to_excel(
                writer, sheet_name="edss_scores", index=False
            )
            pd.DataFrame(
                {
                    "days_after_baseline": relapse_timestamps,
                }
            ).to_excel(writer, sheet_name="relapse_timestamps", index=False)
            # Close the Pandas Excel writer and output the Excel file to the buffer
            writer.close()
            st.download_button(
                key="download_single_follow_up_example",
                label="Download example follow-up and relapse data in .xlsx format",
                data=dl_file_buffer,
                file_name="example_follow_up_data.xlsx",
            )

        st.write("**Upload your own data and select parameters**")
        data_upload_column, option_selection_column, plot_column = st.columns(
            [15, 30, 55]
        )

        with data_upload_column:
            st.write("Upload a follow-up")
            uploaded_single_follow_up = st.file_uploader(
                "Upload your follow-up data as .xlsx", type=["xlsx"]
            )
            if uploaded_single_follow_up is not None:
                raw_follow_up_data = load_excel_table(
                    uploaded_file=uploaded_single_follow_up
                )
                uploaded_single_follow_up_df = pd.read_excel(raw_follow_up_data)

            st.write("Upload relapses (optional)")
            uploaded_single_follow_up_relapses = st.file_uploader(
                "Upload your relapse data as .xlsx", type=["xlsx"]
            )
            if uploaded_single_follow_up_relapses is not None:
                raw_relapses_data = load_excel_table(
                    uploaded_file=uploaded_single_follow_up_relapses
                )
                uploaded_single_follow_up_relapses_df = pd.read_excel(raw_relapses_data)
                uploaded_single_follow_up_relapses_list = list(
                    uploaded_single_follow_up_relapses_df["days_after_baseline"]
                )

            if uploaded_single_follow_up is not None:
                st.write("Preview of uploaded follow-up")
                st.dataframe(uploaded_single_follow_up_df.head())

            if uploaded_single_follow_up_relapses is not None:
                st.write("Preview of uploaded relapses")
                st.dataframe(uploaded_single_follow_up_relapses_df.head())

            if uploaded_single_follow_up_relapses is None:
                uploaded_single_follow_up_relapses_list = []

        with option_selection_column:
            st.write("Select definition options")
            options_for_single_uploaded_example = (
                frontend.dynamic_progression_option_input_element(
                    element_base_key="options_for_single_uploaded_example",
                    default_annotation_mode="symmetric",
                    default_undefined_events_annotation_mode="all",
                    default_baseline="roving",
                    default_confirmation_requirement=True,
                    default_confirmation_duration=30,
                    display_rms_options=True,
                    display_allow_relapses_in_pira_conf=False,
                )
            )

        with plot_column:
            # Instantiate an annotator
            annotator_instance_for_single_uploaded_example = instantiate_annotator(
                # Options
                annotation_mode=options_for_single_uploaded_example["annotation_mode"],
                undefined_events_annotation_mode=options_for_single_uploaded_example[
                    "undefined_events_annotation_mode"
                ],
                opt_raw_before_relapse_max_time=options_for_single_uploaded_example[
                    "opt_raw_before_relapse_max_time"
                ],
                opt_raw_after_relapse_max_time=options_for_single_uploaded_example[
                    "opt_raw_after_relapse_max_time"
                ],
                opt_pira_allow_relapses_between_event_and_confirmation=options_for_single_uploaded_example.get(
                    "opt_pira_allow_relapses_between_event_and_confirmation", False
                ),
                opt_baseline_type=options_for_single_uploaded_example[
                    "opt_baseline_type"
                ],
                opt_roving_reference_require_confirmation=options_for_single_uploaded_example[
                    "opt_roving_reference_require_confirmation"
                ],
                opt_roving_reference_confirmation_time=options_for_single_uploaded_example[
                    "opt_roving_reference_confirmation_time"
                ],
                opt_max_score_that_requires_plus_1=options_for_single_uploaded_example[
                    "opt_increase_threshold"
                ],
                opt_larger_increment_from_0=options_for_single_uploaded_example[
                    "opt_larger_minimal_increase_from_0"
                ],
                opt_minimal_distance_time=options_for_single_uploaded_example[
                    "opt_minimal_distance_time"
                ],
                opt_minimal_distance_type=options_for_single_uploaded_example[
                    "opt_minimal_distance_type"
                ],
                opt_minimal_distance_backtrack_decrease=options_for_single_uploaded_example[
                    "opt_minimal_distance_backtrack_decrease"
                ],
                opt_require_confirmation=options_for_single_uploaded_example[
                    "opt_require_confirmation"
                ],
                opt_confirmation_time=options_for_single_uploaded_example[
                    "opt_confirmation_time"
                ],
                opt_confirmation_type=options_for_single_uploaded_example[
                    "opt_confirmation_type"
                ],
                opt_confirmation_included_values=options_for_single_uploaded_example[
                    "opt_confirmation_included_values"
                ],
                opt_confirmation_sustained_minimal_distance=options_for_single_uploaded_example[
                    "opt_confirmation_sustained_minimal_distance"
                ],
            )

            # Annotate the dataframe
            if uploaded_single_follow_up is not None:
                annotated_uploaded_single_follow_up_df = cached_annotated_df(
                    annotator_instance=annotator_instance_for_single_uploaded_example,
                    follow_up_dataframe=uploaded_single_follow_up_df,
                    relapse_timestamps=uploaded_single_follow_up_relapses_list,
                )

                # Plot it
                fig = figure.Figure(figsize=(16, 6))
                ax = fig.subplots(1)
                visualization.plot_annotated_follow_up(
                    annotated_uploaded_single_follow_up_df,
                    annotation_mode=options_for_single_uploaded_example[
                        "annotation_mode"
                    ],
                    opt_raw_before_relapse_max_time=options_for_single_uploaded_example[
                        "opt_raw_before_relapse_max_time"
                    ],
                    opt_raw_after_relapse_max_time=options_for_single_uploaded_example[
                        "opt_raw_after_relapse_max_time"
                    ],
                    xlabel="Days after baseline",
                    ax=ax,
                )
                fig.tight_layout()
                sns.despine(bottom=True, left=True, right=True, top=True, ax=ax)
                st.pyplot(fig, clear_figure=True)

                # Display overall stats
                cohort_stats_overall_uploaded_single_follow_up_df = (
                    Eval.get_cohort_stats(
                        stats_by_follow_up=Eval.get_follow_up_stats(
                            annotated_follow_ups=annotated_uploaded_single_follow_up_df,
                            get_stats_by_type=False,
                            id_columns=None,
                        ),
                        get_stats_by_type=False,
                        follow_up_id_column=None,
                        groupby_columns=None,
                    )
                )
                cohort_stats_overall_uploaded_single_follow_up_df_display = cohort_stats_overall_uploaded_single_follow_up_df[
                    [
                        "total_events",
                        "total_accrual_events",
                        "total_improvement_events",
                        "total_event_score_delta",
                        "total_accrual_event_score_delta",
                        "total_improvement_event_score_delta",
                        "contribution_of_accrual_to_total_events",
                        "contribution_of_improvement_to_total_events",
                    ]
                ].rename(
                    columns={
                        "total_events": "Total events",
                        "total_accrual_events": "Accrual events",
                        "total_improvement_events": "Improvement events",
                        "total_event_score_delta": "EDSS delta",
                        "total_accrual_event_score_delta": "Accrual EDSS delta",
                        "total_improvement_event_score_delta": "Improvement EDSS delta",
                        "contribution_of_accrual_to_total_events": "Contribution of accrual to total events",
                        "contribution_of_improvement_to_total_events": "Contribution of improvement to total events",
                    }
                )
                st.write("Overall results, scroll to the right for more columns")
                st.dataframe(cohort_stats_overall_uploaded_single_follow_up_df_display)

                # Display stats by event type
                follow_up_stats_uploaded_single_follow_up_df = Eval.get_follow_up_stats(
                    annotated_follow_ups=annotated_uploaded_single_follow_up_df,
                    get_stats_by_type=True,
                    id_columns=None,
                )
                cohort_stats_uploaded_single_follow_up_df = Eval.get_cohort_stats(
                    stats_by_follow_up=follow_up_stats_uploaded_single_follow_up_df,
                    get_stats_by_type=True,
                    follow_up_id_column=None,
                    groupby_columns=None,
                )
                cohort_stats_uploaded_single_follow_up_df_display = cohort_stats_uploaded_single_follow_up_df[
                    [
                        "event_type",
                        "total_events",
                        "total_event_score_delta",
                        "contribution_to_total_events",
                        "contribution_to_total_accrual_events",
                        "contribution_to_total_accrual_delta",
                    ]
                ].rename(
                    columns={
                        "event_type": "Event type",
                        "total_events": "Total events",
                        "total_event_score_delta": "Total EDSS delta",
                        "contribution_to_total_events": "Contribution to total events",
                        "contribution_to_total_accrual_events": "Contribution to accrual events",
                        "contribution_to_total_accrual_delta": "Contribution to accrual delta",
                    }
                )
                st.write("Results by type, scroll to the right for more columns")
                st.dataframe(cohort_stats_uploaded_single_follow_up_df_display)
