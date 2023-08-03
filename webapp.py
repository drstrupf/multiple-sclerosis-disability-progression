"""Streamlit webapp source file.

This is the source code for the Steramlit webapp deployed on
https://multiple-sclerosis-disability-progression.streamlit.app/

"""

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
from evaluation import survival
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


# Function to annotate progression and cache the result
@st.cache_data()
def cache_annotate_progression_for_cohort(
    follow_up_dataframe,
    options_dictionary,
):
    return cached.follow_ups_to_annotated_progression(
        follow_up_dataframe=follow_up_dataframe,
        options_dictionary=options_dictionary,
    )


# Function to compute time to progression and cache the result
@st.cache_data()
def cache_get_lifelines_input_for_cohort(
    follow_up_dataframe,
    options_dictionary,
):
    return cached.follow_ups_to_lifelines_input(
        follow_up_dataframe=follow_up_dataframe,
        options_dictionary=options_dictionary,
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

    st.write("Last cache refresh: " + show_clock_last_cache_refresh())

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
            )

        with option_selection_column:
            st.write("Select definition options")

            options = frontend.dynamic_progression_option_input_element(
                element_base_key="plot_playground_options",
                default_baseline="roving",
            )

            annotated_example_follow_up_df = cache_annotate_progression_for_cohort(
                follow_up_dataframe=edited_example_follow_up_df,
                options_dictionary=options,
            )

        with plot_column:
            st.write(
                "Change definition options to see their influence on the time to first progression."
            )

            visualization.plot_single_annotated_followup(
                annotated_example_follow_up_df=annotated_example_follow_up_df[
                    annotated_example_follow_up_df["follow_up_id"] == 0
                ],
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
                    options_dictionary=options_1,
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
                    options_dictionary=options_2,
                )
            )

        with result_column:
            visualization.plot_kaplan_meier_comparison(
                times_to_event_df_1=sample_follow_ups_time_to_progression_1,
                times_to_event_df_2=sample_follow_ups_time_to_progression_2,
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
