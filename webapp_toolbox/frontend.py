"""Frontend elements for the streamlit webapp."""

import numpy as np
import pandas as pd
import streamlit as st


def baseline_definition_dropdown(key, default_baseline="fixed"):
    """Display a dropdown menu with baseline options and
    return parsed options.

    Args:
        - key: the element's key

    Returns:
        - str: the baseline option
        - bool: confirmation yes/no
        - float: confirmation time for roving reference

    """
    if default_baseline not in ["fixed", "roving"]:
        raise ValueError(
            "Invalid default baseline type. Options are 'fixed' or 'roving'."
        )
    if default_baseline == "fixed":
        options = ["Fixed baseline", "Roving reference"]
    elif default_baseline == "roving":
        options = ["Roving reference", "Fixed baseline"]
    option_baseline_type = st.selectbox(
        label="Baseline type",
        options=options,
        key=key,
    )
    if option_baseline_type == "Fixed baseline":
        return "fixed", False, 0
    elif option_baseline_type == "Roving reference":
        option_roving_require_confirmation = st.selectbox(
            label="Require confirmation of roving reference?",
            options=[
                "No confirmation required",
                "Yes, for a specific confirmation time",
            ],
            key=key + "_roving_require_confirmation",
        )
        if option_roving_require_confirmation == "No confirmation required":
            option_roving_require_confirmation = False
        else:
            option_roving_require_confirmation = True
        if option_roving_require_confirmation:
            opt_roving_reference_confirmation_time = st.number_input(
                label="Confirmation time for roving reference",
                min_value=1,
                value=1,
                key=key + "_roving_confirmation_time",
            )
        else:
            opt_roving_reference_confirmation_time = 0
        return "roving", option_roving_require_confirmation, opt_roving_reference_confirmation_time


def minimum_required_increase_threshold_dropdown(key):
    """Display a dropdown menu for choosing the minimal
    EDSS increase for progression, and return a parsed
    minimal increase threshold.

    Args:
        - key: the element's key

    Returns
        - float: the minimum increase threshold

    """
    option_minimal_increase_threshold = st.selectbox(
        label="Minimal EDSS score increase for progression",
        options=[
            "+ 1.0 for reference ≤ 5.0, + 0.5 else",
            "+ 0.5",
            "+ 1.0",
            "+ 1.0 for reference ≤ 1.0, + 0.5 else",
            "+ 1.0 for reference ≤ 1.5, + 0.5 else",
            "+ 1.0 for reference ≤ 2.0, + 0.5 else",
            "+ 1.0 for reference ≤ 2.5, + 0.5 else",
            "+ 1.0 for reference ≤ 3.0, + 0.5 else",
            "+ 1.0 for reference ≤ 3.5, + 0.5 else",
            "+ 1.0 for reference ≤ 4.0, + 0.5 else",
            "+ 1.0 for reference ≤ 4.5, + 0.5 else",
            "+ 1.0 for reference ≤ 5.0, + 0.5 else",
            "+ 1.0 for reference ≤ 5.5, + 0.5 else",
            "+ 1.0 for reference ≤ 6.0, + 0.5 else",
            "+ 1.0 for reference ≤ 6.5, + 0.5 else",
            "+ 1.0 for reference ≤ 7.0, + 0.5 else",
            "+ 1.0 for reference ≤ 7.5, + 0.5 else",
            "+ 1.0 for reference ≤ 8.0, + 0.5 else",
            "+ 1.0 for reference ≤ 8.5, + 0.5 else",
            "+ 1.0 for reference ≤ 9.0, + 0.5 else",
            "+ 1.0 for reference ≤ 9.5, + 0.5 else",
        ],
        key=key,
    )
    if option_minimal_increase_threshold.startswith("+ 1.0 for reference ≤"):
        option_minimal_increase_threshold = float(
            option_minimal_increase_threshold.split("≤ ")[1]
            .split(" ")[0]
            .replace(",", "")
            .strip()
        )
    elif option_minimal_increase_threshold == "+ 0.5":
        option_minimal_increase_threshold = 0.0
    elif option_minimal_increase_threshold == "+ 1.0":
        option_minimal_increase_threshold = 10.0
    return option_minimal_increase_threshold


def larger_increase_from_0_dropdown(key):
    """Display a dropdown menu for choosing whether the
    increase from EDSS 0 has to be 1.5 or not.

    Args:
        - key: the element's key

    Returns
        - bool: whether increase from 0 has to be 1.5

    """
    option_larger_from_0 = st.selectbox(
        label="Require + 1.5 if reference EDSS is 0?",
        options=["Yes", "No"],
        key=key,
    )
    if option_larger_from_0 == "Yes":
        return True
    elif option_larger_from_0 == "No":
        return False


def minimal_distance_requirement_dropdown(key):
    """Display a collection of dropdown menues
    for choosing the minimal minimal distance condition.

    Depending on the input, more or less fields are displayed
    to prevent the user from selecting invalid option combos.

    Args:
        - key: the element's key

    Returns
        - str: option_minimal_distance_type - the minimal distance type (none, previous, reference)
        - int: option_minimal_distance_time - the minimal distance duration
        - bool: option_minimal_distance_backtrack_monotonic_decrease - for 'reference', adjust for monotonic decrease

    """
    min_distance_input = st.selectbox(
        label="Minimal distance requirement",
        options=[
            "No minimal distance requirement",
            "Minimal distance to reference assessment",
            "Minimal distance to previous assessment",
        ],
        key=key + "_type",
    )
    if min_distance_input == "No minimal distance requirement":
        option_minimal_distance_type = "reference"
        option_minimal_distance_time = 0
    else:
        if min_distance_input == "Minimal distance to reference assessment":
            option_minimal_distance_type = "reference"
        elif min_distance_input == "Minimal distance to previous assessment":
            option_minimal_distance_type = "previous"
        option_minimal_distance_time = st.number_input(
            label="Minimal distance",
            min_value=0,
            value=0,
            key=key + "_time",
        )
    option_minimal_distance_backtrack_monotonic_decrease = False
    if min_distance_input == "Minimal distance to reference assessment":
        option_minimal_distance_backtrack_monotonic_decrease = st.selectbox(
            label="Adjust for monotonic decrease when computing distance to reference?",
            options=["Yes", "No"],
            key=key + "_backtrack_monotonic_decrease",
        )
        if option_minimal_distance_backtrack_monotonic_decrease == "Yes":
            option_minimal_distance_backtrack_monotonic_decrease = True
        elif option_minimal_distance_backtrack_monotonic_decrease == "No":
            option_minimal_distance_backtrack_monotonic_decrease = False
    return (
        option_minimal_distance_type,
        option_minimal_distance_time,
        option_minimal_distance_backtrack_monotonic_decrease,
    )


def confirmation_requirement_dropdown(key):
    """Display a collection of dropdown menues
    for choosing the confirmation requirements.

    Depending on the input, more or less fields are displayed
    to prevent the user from selecting invalid option combos.

    Args:
        - key: the element's key

    Returns
        - bool: option_require_confirmation - whether confirmation is required
        - int: option_confirmation_time - duration of confirmation interval
        - str: option_confirmation_included_values - which values to consider
        - str: option_confirmation_type - confirmation condition type

    """
    option_require_confirmation = st.selectbox(
        label="Require confirmation?",
        options=[
            "No confirmation required",
            "Yes, for a specific confirmation time",
            "Yes, sustained over the entire follow-up",
        ],
        key=key + "_require_confirmation",
    )
    if option_require_confirmation == "Yes, for a specific confirmation time":
        option_require_confirmation = True
        option_confirmation_sustained_minimal_distance = 0
        option_confirmation_time = st.number_input(
            label="Confirmation time",
            min_value=1,
            value=12,
            key=key + "_confirmation_time",
        )
        option_confirmation_included_values = st.selectbox(
            label="Scores relevant for confirmation",
            options=[
                "All values within confirmation interval",
                "Only the last value in confirmation interval",
            ],
            key=key + "_confirmation_included_values",
        )
        if (
            option_confirmation_included_values
            == "All values within confirmation interval"
        ):
            option_confirmation_included_values = "all"
        elif (
            option_confirmation_included_values
            == "Only the last value in confirmation interval"
        ):
            option_confirmation_included_values = "last"

    elif option_require_confirmation == "Yes, sustained over the entire follow-up":
        option_require_confirmation = True
        option_confirmation_time = -1
        option_confirmation_included_values = "all"
        option_confirmation_sustained_minimal_distance = st.number_input(
            label="Minimal duration for sustained",
            min_value=0,
            value=0,
            key=key + "_sustained_minimal_distance",
        )

    elif option_require_confirmation == "No confirmation required":
        option_require_confirmation = False
        option_confirmation_time = -1
        option_confirmation_included_values = "all"
        option_confirmation_type = "minimum"
        option_confirmation_sustained_minimal_distance = 0

    if option_require_confirmation:
        option_confirmation_type = st.selectbox(
            label="Confirmation condition",
            options=[
                "All values ≥ reference + minimal increase",
                "All values ≥ progression candidate score",
            ],
            key=key + "_confirmation_type",
        )
        if option_confirmation_type == "All values ≥ reference + minimal increase":
            option_confirmation_type = "minimum"
        elif option_confirmation_type == "All values ≥ progression candidate score":
            option_confirmation_type = "monotonic"

    return (
        option_require_confirmation,
        option_confirmation_time,
        option_confirmation_included_values,
        option_confirmation_type,
        option_confirmation_sustained_minimal_distance,
    )

def undefined_progression_dropdown(key, default="re-baselining only"):
    option_undefined_progression = st.selectbox(
        label="Undefined progression option",
        options=[
            "Re-baselining only",
            "Never",
            "All",
            "End",
        ],
        key=key + "_require_confirmation",
    )
    option_undefined_progression = option_undefined_progression.lower()
    return option_undefined_progression


def dynamic_progression_option_input_element(element_base_key, default_baseline, display_rms_options=False):
    """Creates a series of input widgets and returns the options as dict.

    This frontend element creates a set of dropdown options to select
    baseline type, minimal increase, minimal distance, and confirmation
    requirements, and returns a dict with the chosen options.

    Dropdown elements are only displayed when relevant, as implemented in
    the respective element's code, such that the user can not choose
    invalid parameter combinations.

    Args:
        - element_base_key: the element's base key, a str
        - default_baseline: the baseline type to be displayed at first load

    Returns:
        - dict: a dictionary with the options for annotating the baseline
          and first progression events.

    """
    rms_options = {}
    if display_rms_options:
        # Undefined progression
        undefined_progression = undefined_progression_dropdown(key=element_base_key + "_undefined_progression", default="re-baselining only")
        rms_options = {"undefined_progression": undefined_progression,}
    # Baseline type
    option_baseline_type, baseline_confirmation, baseline_confirmation_distance = baseline_definition_dropdown(
        key=element_base_key + "_option_baseline_type",
        default_baseline=default_baseline,
    )
    # Increase threshold
    option_minimal_increase_threshold = minimum_required_increase_threshold_dropdown(
        key=element_base_key + "_option_minimal_increase_threshold"
    )
    # Larger increase from 0
    option_larger_increase_from_0 = larger_increase_from_0_dropdown(
        key=element_base_key + "_option_larger_increase_from_0"
    )
    # Confirmation requirements
    (
        option_require_confirmation,
        option_confirmation_time,
        option_confirmation_included_values,
        option_confirmation_type,
        option_confirmation_sustained_minimal_distance,
    ) = confirmation_requirement_dropdown(key=element_base_key + "_option_confirmation")
    # Minimal distance requirements
    (
        option_minimal_distance_type,
        option_minimal_distance_time,
        option_minimal_distance_backtrack_monotonic_decrease,
    ) = minimal_distance_requirement_dropdown(
        key=element_base_key + "_option_minimal_distance"
    )
    return {**{
        "opt_baseline_type": option_baseline_type,
        "opt_roving_reference_require_confirmation": baseline_confirmation,
        "opt_roving_reference_confirmation_time": baseline_confirmation_distance,
        "opt_increase_threshold": option_minimal_increase_threshold,
        "opt_larger_minimal_increase_from_0": option_larger_increase_from_0,
        "opt_minimal_distance_time": option_minimal_distance_time,
        "opt_minimal_distance_type": option_minimal_distance_type,
        "opt_minimal_distance_backtrack_decrease": option_minimal_distance_backtrack_monotonic_decrease,
        "opt_require_confirmation": option_require_confirmation,
        "opt_confirmation_time": option_confirmation_time,
        "opt_confirmation_type": option_confirmation_type,
        "opt_confirmation_included_values": option_confirmation_included_values,
        "opt_confirmation_sustained_minimal_distance": option_confirmation_sustained_minimal_distance,
    }, **rms_options}


def example_input_dataframe_editor(
    follow_up_dataframe,
    element_base_key,
    edss_score_column_name="edss_score",
    time_column_name="days_after_baseline",
    id_column_name="follow_up_id",
    display_id_column=False,
):
    """TBD; source for session state hack:
    https://discuss.streamlit.io/t/undo-changes-in-st-experimental-data-editor/42914/9
    """
    if element_base_key not in st.session_state:
        st.session_state[element_base_key] = 0

    edited_follow_up_dataframe = pd.DataFrame(
        {
            id_column_name: list(follow_up_dataframe[id_column_name]),
            time_column_name: list(follow_up_dataframe[time_column_name]),
            edss_score_column_name: list(follow_up_dataframe[edss_score_column_name]),
        }
    )

    def _reset_example_follow_up_editor():
        st.session_state[element_base_key] += 1

    if display_id_column:
        columns_to_display = [id_column_name, time_column_name, edss_score_column_name]
    else:
        columns_to_display = [time_column_name, edss_score_column_name]

    edited_follow_up_dataframe = st.data_editor(
        edited_follow_up_dataframe,
        column_config={
            id_column_name: "ID",
            time_column_name: "Time",
            edss_score_column_name: st.column_config.SelectboxColumn(
                "EDSS", help="EDSS score", options=[i * 0.5 for i in range(21)]
            ),
        },
        hide_index=True,
        disabled=[id_column_name, time_column_name],
        column_order=columns_to_display,
        key=element_base_key + f"editor_{st.session_state[element_base_key]}",
    )

    st.button("Reset example data", on_click=_reset_example_follow_up_editor, key=element_base_key + "_reset_button")

    return edited_follow_up_dataframe


if __name__ == "__main__":
    pass
