import numpy as np
import pandas as pd
import streamlit as st
import xlsxwriter
from io import BytesIO
from datetime import date
from datetime import datetime
from datetime import timezone

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
sns.set_style("whitegrid", {"grid.color": "gainsboro"})

from definitions import baselines, progression
from evaluation import utils, survival
from webapp_toolbox import frontend, visualization, cached

# Wide layout
st.set_page_config(layout="wide")


# Cached clock to help keeping track of cache misses
@st.cache_data()
def show_clock_last_cache_refresh():
    return (
        datetime.today().strftime("%d.%m.%Y, %H:%M:%S")
        + " "
        + str(datetime.now(timezone.utc).astimezone().tzinfo)
    )


# Load and cache sampled cohort data
@st.cache_data()
def load_cache_sampled_data():
    return pd.read_excel("data/examples/sampled_follow_ups.xlsx")


# Function to compute time to progression and cache the result
@st.cache_data()
def cache_get_lifelines_input_for_cohort(
    follow_up_dataframe,
    baseline_type="fixed",
    opt_increase_threshold=5.5,
    opt_larger_minimal_increase_from_0=True,
    opt_minimal_distance_time=0,
    opt_minimal_distance_type="reference",
    opt_minimal_distance_backtrack_monotonic_decrease=True,
    opt_require_confirmation=False,
    opt_confirmation_time=-1,
    opt_confirmation_type="minimum",
    opt_confirmation_included_values="all",
    id_column_name="follow_up_id",
    edss_score_column_name="edss_score",
    time_column_name="days_after_baseline",
):
    return cached.get_lifelines_input_for_cohort(
        follow_up_dataframe=follow_up_dataframe,
        baseline_type=baseline_type,
        opt_increase_threshold=opt_increase_threshold,
        opt_larger_minimal_increase_from_0=opt_larger_minimal_increase_from_0,
        opt_minimal_distance_time=opt_minimal_distance_time,
        opt_minimal_distance_type=opt_minimal_distance_type,
        opt_minimal_distance_backtrack_monotonic_decrease=opt_minimal_distance_backtrack_monotonic_decrease,
        opt_require_confirmation=opt_require_confirmation,
        opt_confirmation_time=opt_confirmation_time,
        opt_confirmation_type=opt_confirmation_type,
        opt_confirmation_included_values=opt_confirmation_included_values,
        id_column_name=id_column_name,
        edss_score_column_name=edss_score_column_name,
        time_column_name=time_column_name,
    )


# Example follow-up
example_follow_up_df = pd.DataFrame(
    [
        {"follow_up_id": 0, "days_after_baseline": 0, "edss_score": 5.0},
        {"follow_up_id": 0, "days_after_baseline": 12, "edss_score": 4.5},
        {"follow_up_id": 0, "days_after_baseline": 24, "edss_score": 4.5},
        {"follow_up_id": 0, "days_after_baseline": 36, "edss_score": 5.0},
        {"follow_up_id": 0, "days_after_baseline": 48, "edss_score": 4.5},
        {"follow_up_id": 0, "days_after_baseline": 60, "edss_score": 4.0},
        {"follow_up_id": 0, "days_after_baseline": 72, "edss_score": 3.5},
        {"follow_up_id": 0, "days_after_baseline": 84, "edss_score": 5.5},
        {"follow_up_id": 0, "days_after_baseline": 96, "edss_score": 5.5},
        {"follow_up_id": 0, "days_after_baseline": 108, "edss_score": 4.5},
    ]
)

if __name__ == "__main__":
    # Setup general app layout
    st.title("Disability Progression in Multiple Sclerosis from EDSS Scores")
    st.markdown(
        "A little playground to explore the myriads of possible definitions of *first disability progression*."
    )

    # More detailed author information
    with st.expander("Author information and contact", expanded=False):
        st.markdown("### Authors")
        st.markdown(
            r"**Gabriel Bsteh**<sup>1, 2</sup>, **Stefanie Marti**<sup>3</sup>, **Robert Hoepner**<sup>3</sup>",
            unsafe_allow_html=True,
        )
        st.markdown(
            r"""<sup>1</sup>Department of Neurology, Medical University of Vienna, Vienna, Austria  
        <sup>2</sup>Comprehensive Center for Clinical Neurosciences and Mental Health, Medical University of Vienna, Vienna, Austria  
        <sup>3</sup>Department of Neurology, Inselspital, Bern University Hospital and University of Bern, Switzerland""",
            unsafe_allow_html=True,
        )
        st.markdown("### Contact information")
        st.markdown(
            r"""Found a **bug**? Do you have a **feature request**? We would appreciate your feedback!
            Please open an issue on our [GitHub project page](https://github.com/drstrupf/multiple-sclerosis-disability-progression)."""
        )
        st.markdown(r"TBD: corresponding authors, citation, license...")

    # Add some hints/FAQ
    with st.expander("References", expanded=False):
        st.markdown(
            r"""[1] Bsteh G, Marti S, Hegen H, Krajnc N, Traxler G, Hammer H,
            Leutmezer F, Rommer P, Di Pauli F, Chan A, Berger T, Hoepner R.
            **Disability progression is a question of definition - a methodological
            reappraisal by example of primary progressive multiple sclerosis**. TBD."""
        )
        st.markdown(
            r"""[2] Kappos L, Butzkueven H, Wiendl H, Spelman T, Pellegrini F,
            Chen Y, Dong Q, Koendgen H, Belachew S, Trojano M; Tysabri® 
            Observational Program (TOP) Investigators. **Greater sensitivity 
            to multiple sclerosis disability worsening and progression 
            events using a roving versus a fixed reference value in a 
            prospective cohort study**. Multiple Sclerosis Journal
            24.7 (2017), pp. 963-973. doi: 10.1177/1352458517709619"""
        )

    st.write(
        # ":watch: Radio reloj: "
        "Last cache refresh: "
        + show_clock_last_cache_refresh()
        # + " :zap: :skull_and_crossbones: :fire: :boom: :skull:"
    )

    st.write("## Explore definition options")

    with st.expander(
        "Plot follow-up and annotate first progression event for example data",
        expanded=True,
    ):
        data_edit_column, option_selection_column, plot_column = st.columns(
            [15, 40, 45]
        )

        with data_edit_column:
            st.write("Edit example dataframe")

            edited_example_follow_up_df = frontend.example_input_dataframe_editor(
                follow_up_dataframe=example_follow_up_df,
                element_base_key="example_follow_up_editor_key",
                edss_score_column_name="edss_score",
                time_column_name="days_after_baseline",
                id_column_name="follow_up_id",
                display_id_column=False,
            )

        with option_selection_column:
            st.write("Select definition options")

            annotated_example_follow_up_df = (
                frontend.annotate_first_progression_to_follow_up_dynamic_element(
                    follow_up_dataframe=edited_example_follow_up_df,
                    element_base_key="plot_playground_options",
                )
            )

        with plot_column:
            st.write(
                "Change definition options to see their influence on the time to first progression."
            )

            visualization.plot_single_annotated_followup(
                annotated_example_follow_up_df=annotated_example_follow_up_df[
                    annotated_example_follow_up_df["follow_up_id"] == 0
                ],
                edss_score_column_name="edss_score",
                time_column_name="days_after_baseline",
                reference_score_column_name="reference_edss_score",
                first_progression_flag_column_name="is_first_progression",
                figsize=(12, 8),
            )

    st.write("## Compare definitions for an example cohort")

    with st.expander(
        "Display a Kaplan-Meier graph for two progression definitions for a set of 200 randomly generated follow-ups.",
        expanded=True,
    ):
        sample_follow_ups = load_cache_sampled_data()

        (
            options_title_column,
            result_title_column,
        ) = st.columns([55, 45])

        with options_title_column:
            st.write("### Choose two definitions to compare")
        with result_title_column:
            st.write("### Compare Kaplan-Meier estimates")

        (
            first_option_selection_column,
            second_option_selection_column,
            result_column,
        ) = st.columns([27.5, 27.5, 45])

        with first_option_selection_column:
            st.write("Select first set of definition options.")

            options_1 = frontend.dynamic_progression_option_input_element(
                element_base_key="example_cohort_options_form_contents_def_1",
                default_baseline="fixed",
            )
            sample_follow_ups_time_to_progression_1 = (
                cache_get_lifelines_input_for_cohort(
                    follow_up_dataframe=sample_follow_ups,
                    baseline_type=options_1["baseline_type"],
                    opt_increase_threshold=options_1["opt_increase_threshold"],
                    opt_larger_minimal_increase_from_0=options_1[
                        "opt_larger_minimal_increase_from_0"
                    ],
                    opt_minimal_distance_time=options_1["opt_minimal_distance_time"],
                    opt_minimal_distance_type=options_1["opt_minimal_distance_type"],
                    opt_minimal_distance_backtrack_monotonic_decrease=options_1[
                        "opt_minimal_distance_backtrack_monotonic_decrease"
                    ],
                    opt_require_confirmation=options_1["opt_require_confirmation"],
                    opt_confirmation_time=options_1["opt_confirmation_time"],
                    opt_confirmation_type=options_1["opt_confirmation_type"],
                    opt_confirmation_included_values=options_1[
                        "opt_confirmation_included_values"
                    ],
                )
            )

        with second_option_selection_column:
            st.write("Select second set of definition options.")

            options_2 = frontend.dynamic_progression_option_input_element(
                element_base_key="example_cohort_options_form_contents_def_2",
                default_baseline="roving",
            )
            sample_follow_ups_time_to_progression_2 = (
                cache_get_lifelines_input_for_cohort(
                    follow_up_dataframe=sample_follow_ups,
                    baseline_type=options_2["baseline_type"],
                    opt_increase_threshold=options_2["opt_increase_threshold"],
                    opt_larger_minimal_increase_from_0=options_2[
                        "opt_larger_minimal_increase_from_0"
                    ],
                    opt_minimal_distance_time=options_2["opt_minimal_distance_time"],
                    opt_minimal_distance_type=options_2["opt_minimal_distance_type"],
                    opt_minimal_distance_backtrack_monotonic_decrease=options_2[
                        "opt_minimal_distance_backtrack_monotonic_decrease"
                    ],
                    opt_require_confirmation=options_2["opt_require_confirmation"],
                    opt_confirmation_time=options_2["opt_confirmation_time"],
                    opt_confirmation_type=options_2["opt_confirmation_type"],
                    opt_confirmation_included_values=options_2[
                        "opt_confirmation_included_values"
                    ],
                )
            )

        with result_column:
            visualization.plot_kaplan_meier_comparison(
                times_to_event_df_1=sample_follow_ups_time_to_progression_1,
                times_to_event_df_2=sample_follow_ups_time_to_progression_2,
                durations_column_name="duration",
                observed_column_name="observed",
                figsize=(12, 8),
                xlim=None,
            )

        # Give some hints
        (
            instructions_column,
            download_column,
        ) = st.columns([55, 45])

        with instructions_column:
            st.info(
                "### Hint\n\nThe example data used here use *days after baseline* as time unit. "
                "Thus, if you want to require a minimum number of weeks for confirmation or "
                "a minimal distance to the reference/previous assessment, you have to "
                "multiply the number of weeks by 7 when filling the corresponding time fields.\n\n"
                "**Example:** For a duration of 12 weeks, enter '84' in the respective field.\n\n"
                "12 weeks = 84 days, 24 weeks = 168 days, and 48 weeks = 336 days."
            )
        with download_column:
            st.write("Download example data")
            download_file_buffer = BytesIO()
            with pd.ExcelWriter(download_file_buffer, engine="xlsxwriter") as writer:
                sample_follow_ups.to_excel(
                    writer, sheet_name="SampleFollowUps", index=False
                )
                writer.close()
                st.download_button(
                    label="Download sample data .xlsx",
                    data=download_file_buffer,
                    file_name="sample_follow_up_data.xlsx",
                )