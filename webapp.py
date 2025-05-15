"""Streamlit webapp source file.

This is the source code for the Steramlit webapp deployed on
https://multiple-sclerosis-disability-progression.streamlit.app/

"""

import gc
import numpy as np
import pandas as pd
import streamlit as st
import xlsxwriter
from io import BytesIO
from datetime import date
from datetime import datetime
from datetime import timezone

import matplotlib

matplotlib.use("Agg")
from matplotlib import figure
import seaborn as sns

sns.set_theme(color_codes=True)
sns.set_style("whitegrid", {"grid.color": "gainsboro"})

from webapp_toolbox import frontend
from definitions import edssprogression
from tools import visualization

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


# Cached EDSS disability accrual annotation instance
@st.cache_data(ttl=60)
def instantiate_annotator(
    undefined_progression="re-baselining only",
    undefined_progression_wrt_raw_pira_baseline="any",
    opt_baseline_type="roving",
    opt_roving_reference_require_confirmation=True,
    opt_roving_reference_confirmation_time=30,
    opt_roving_reference_confirmation_included_values="all",
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
    opt_minimal_distance_time=0,
    opt_minimal_distance_type="reference",
    opt_minimal_distance_backtrack_decrease=True,
):
    return edssprogression.EDSSProgression(
        undefined_progression=undefined_progression,
        undefined_progression_wrt_raw_pira_baseline=undefined_progression_wrt_raw_pira_baseline,
        opt_baseline_type=opt_baseline_type,
        opt_roving_reference_require_confirmation=opt_roving_reference_require_confirmation,
        opt_roving_reference_confirmation_time=opt_roving_reference_confirmation_time,
        opt_roving_reference_confirmation_included_values=opt_roving_reference_confirmation_included_values,
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
        opt_minimal_distance_time=opt_minimal_distance_time,
        opt_minimal_distance_type=opt_minimal_distance_type,
        opt_minimal_distance_backtrack_decrease=opt_minimal_distance_backtrack_decrease,
    )


# Cached annotated dataframe
@st.cache_data(ttl=60)
def cached_annotated_df(annotator_instance, follow_up_dataframe, relapse_timestamps):
    return annotator_instance.add_progression_events_to_follow_up(
        follow_up_dataframe=follow_up_dataframe, relapse_timestamps=relapse_timestamps
    )


# Example follow-ups
example_follow_up_df = pd.DataFrame(
    [
        {"days_after_baseline": 0, "edss_score": 1.0},
        {"days_after_baseline": 60, "edss_score": 2.5},
        {"days_after_baseline": 90, "edss_score": 1.5},
        {"days_after_baseline": 150, "edss_score": 1.5},
        {"days_after_baseline": 220, "edss_score": 3.5},
        {"days_after_baseline": 250, "edss_score": 2.0},
        {"days_after_baseline": 310, "edss_score": 2.0},
        {"days_after_baseline": 370, "edss_score": 2.0},
        {"days_after_baseline": 430, "edss_score": 1.5},
        {"days_after_baseline": 490, "edss_score": 1.5},
        {"days_after_baseline": 550, "edss_score": 3.5},
        {"days_after_baseline": 580, "edss_score": 2.5},
        {"days_after_baseline": 640, "edss_score": 2.5},
        {"days_after_baseline": 700, "edss_score": 3.0},
        {"days_after_baseline": 760, "edss_score": 3.5},
        {"days_after_baseline": 820, "edss_score": 4.0},
    ]
)

if __name__ == "__main__":
    # Setup general app layout
    st.title("Disability Progression in Multiple Sclerosis from EDSS Scores")
    st.markdown(
        "A little playground to explore the myriads of possible definitions of *EDSS disability accrual* "
        + "as implemented in [multiple-sclerosis-disability-progression](https://github.com/drstrupf/multiple-sclerosis-disability-progression)."
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
            Please **open an issue** on our [GitHub project page](https://github.com/drstrupf/multiple-sclerosis-disability-progression) or
            **contact the authors** ([Gabriel Bsteh](https://www.meduniwien.ac.at/web/forschung/researcher-profiles/researcher-profiles/detail/?res=gabriel_bsteh&cHash=0896fd3f091c51c7c5c37b55b83d8def),
            [Robert Hoepner](http://www.neurologie.insel.ch/de/ueber-uns/teams/details/person/detail/robert-hoepner))."""
        )

    # Cite the PPMS paper
    with st.expander(
        "Related publication - **Disability progression is a question of definition - A methodological reappraisal by example of primary progressive multiple sclerosis**",
        expanded=False,
    ):
        st.markdown("##### Abstract")
        st.markdown(
            "**Background**: Different definitions of disability progression by Expanded Disability Status Scale (EDSS) may influence frequency and/or time to event.  "
            + "\n\n**Methods**: In this multicenter cohort study, we included PPMS patients with follow-up ≥24 months and ≥3 available EDSS scores overall (≥1 per year).  "
            + "\nWe applied 672 definitions of disability progression including different minimal EDSS increase, required confirmation and fixed/roving-baseline score.  "
            + "\n\n**Results**: We analyzed follow-up periods from 131 PPMS patients (median age at baseline 53.0 years [45.0 - 63.0], 51.9 % female, median follow-up 3.9  "
            + "\nyears [2.6 - 6.0], median baseline EDSS 4.0 [2.5 - 6.0]). The most sensitive definition of a progression event was an unconfirmed increase of ≥0.5 points   "
            + "\nwith a roving baseline (81.8 % event rate). The least sensitive definition was an increase of ≥1.0 points with a fixed baseline, minimal distance to reference  "
            + "\n48 weeks, and confirmed at ≥48 weeks (28.4 % event rate). Comparing roving vs. fixed baseline over all cutoffs and confirmation definitions, average time  "
            + "\nto progression was 227 days shorter applying the roving baseline (1405 days [550 - 2653] vs. 1632 days [760 - 2653]).  "
            + "\n\n**Conclusions**: Different definitions of disability progression result in significantly differing rates of disability progression, which may influence   "
            + "\nstudy results and create confusion in clinical practice."
        )
        st.markdown("##### Reference")
        st.markdown(
            "Bsteh G, Marti S, Krajnc N, Traxler G, Salmen A, Hammer H, Leutmezer F, Rommer P, Di Pauli F, Chan A, Berger T, Hegen H, Hoepner R.  "
            + "\n**Disability progression is a question of definition - A methodological reappraisal by example of primary progressive multiple sclerosis.**  "
            + "\n*Mult Scler Relat Disord*. 2025 Jan;93:106215. doi: 10.1016/j.msard.2024.106215. Epub 2024 Dec 6. [PMID: 39662164](https://pubmed.ncbi.nlm.nih.gov/39662164/)."
        )

    st.write("Last cache refresh: " + show_clock_last_cache_refresh())

    # Expander with the example for playing around
    st.write("## Explore PIRA and RAW definition options")
    st.markdown(
        "This section illustrates how **relapse-related options** as well as **general progression options** affect the number of "
        + " events, the times to event, and the event types."
    )

    with st.expander(
        "Plot follow-up and annotate progression events for example data",
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
                default=[40, 200, 530],
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
                undefined_progression=options_example["undefined_progression"],
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
