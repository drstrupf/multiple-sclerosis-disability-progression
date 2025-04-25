"""Test edssprogression.py functionality.

Execute

    python -m tests.tests

in parent to run tests!

This is a bit messy for now. Use tools.visualization.py for some
visual testing.

"""

import numpy as np
import pandas as pd

from collections import Counter

from definitions import edssprogression


LABEL_UNDEFINED_PROGRESSION = "Undefined"
LABEL_PIRA = "PIRA"
LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW = "PIRA confirmed in RAW window"
LABEL_RAW = "RAW"


# ------------------------
# Part 1 - building blocks
# -------------------------


def test_is_above_progress_threshold():
    # Test 1 - minimum required increase + 0.5 irrespective of reference
    test_cases_1 = [
        {"current": 0, "reference": 0, "target": False},
        {"current": 0.5, "reference": 0, "target": True},
        {"current": 1.0, "reference": 0, "target": True},
        {"current": 1.5, "reference": 0, "target": True},
        {"current": 2.0, "reference": 2.0, "target": False},
        {"current": 2.5, "reference": 2.0, "target": True},
        {"current": 3.0, "reference": 2.0, "target": True},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssprogression.EDSSProgression(
                    opt_max_score_that_requires_plus_1=-1,
                    opt_larger_increment_from_0=False,
                )._is_above_progress_threshold(
                    current_edss=test_case["current"],
                    reference_edss=test_case["reference"],
                )
                for test_case in test_cases_1
            ]
        ),
        np.array([test_case["target"] for test_case in test_cases_1]),
        err_msg="Minimal increase + 0.5 irrespective of baseline failed!",
    )
    # Test 2 - minimum required increase + 1.0 irrespective of reference
    test_cases_2 = [
        {"current": 0, "reference": 0, "target": False},
        {"current": 0.5, "reference": 0, "target": False},
        {"current": 1.0, "reference": 0, "target": True},
        {"current": 1.5, "reference": 0, "target": True},
        {"current": 2.0, "reference": 2.0, "target": False},
        {"current": 2.5, "reference": 2.0, "target": False},
        {"current": 3.0, "reference": 2.0, "target": True},
        {"current": 10.0, "reference": 9.5, "target": False},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssprogression.EDSSProgression(
                    opt_max_score_that_requires_plus_1=10.0,
                    opt_larger_increment_from_0=False,
                )._is_above_progress_threshold(
                    current_edss=test_case["current"],
                    reference_edss=test_case["reference"],
                )
                for test_case in test_cases_2
            ]
        ),
        np.array([test_case["target"] for test_case in test_cases_2]),
        err_msg="Minimal increase + 1.0 irrespective of baseline failed!",
    )
    # Test 3 - minimum required increase from 0 + 1.5
    test_cases_3 = [
        {"current": 0, "reference": 0, "target": False},
        {"current": 0.5, "reference": 0, "target": False},
        {"current": 1.0, "reference": 0, "target": False},
        {"current": 1.5, "reference": 0, "target": True},
        {"current": 2.0, "reference": 2.0, "target": False},
        {"current": 2.5, "reference": 2.0, "target": True},
        {"current": 3.0, "reference": 2.0, "target": True},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssprogression.EDSSProgression(
                    opt_max_score_that_requires_plus_1=1,
                    opt_larger_increment_from_0=True,
                )._is_above_progress_threshold(
                    current_edss=test_case["current"],
                    reference_edss=test_case["reference"],
                )
                for test_case in test_cases_3
            ]
        ),
        np.array([test_case["target"] for test_case in test_cases_3]),
        err_msg="Minimal increase + 1.f from 0 failed!",
    )
    # Test 4 - standard case
    test_cases_4 = [
        {"current": 0, "reference": 0, "target": False},
        {"current": 0.5, "reference": 0, "target": False},
        {"current": 1.0, "reference": 0, "target": False},
        {"current": 1.5, "reference": 0, "target": True},
        {"current": 2.0, "reference": 2.0, "target": False},
        {"current": 2.5, "reference": 2.0, "target": False},
        {"current": 3.0, "reference": 2.0, "target": True},
        {"current": 5.5, "reference": 5.0, "target": False},
        {"current": 6.0, "reference": 5.0, "target": True},
        {"current": 6.0, "reference": 5.5, "target": True},
        {"current": 6.5, "reference": 5.5, "target": True},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssprogression.EDSSProgression(
                    opt_max_score_that_requires_plus_1=5,
                    opt_larger_increment_from_0=True,
                )._is_above_progress_threshold(
                    current_edss=test_case["current"],
                    reference_edss=test_case["reference"],
                )
                for test_case in test_cases_4
            ]
        ),
        np.array([test_case["target"] for test_case in test_cases_4]),
        err_msg="Standard minimal increase settings failed!",
    )


def test_get_confirmation_scores_dataframe():
    test_dataframe = pd.DataFrame(
        {"timestamp": [0, 10, 20, 30, 40, 50], "score": [0, 1, 2, 3, 4, 5]}
    )
    # Test case 1 - sustained, assessments available
    # Must yield all assessments following the first one.
    test_case_1 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=0,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=-1,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=0,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_1.equals(test_dataframe.iloc[1:]), "Test 1 failed!"
    # Test case 2 - sustained, no assessments available
    # Must yield an empty dataframe.
    test_case_2 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=50,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=-1,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=0,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_2.equals(
        test_dataframe[test_dataframe["timestamp"] > 50]
    ), "Test 2 failed!"
    # Test case 3 - sustained, minimal time interval
    # Must yield all assessments following the second one.
    test_case_3 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=0,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=-1,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=15,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_3.equals(test_dataframe.iloc[2:]), "Test 3 failed!"
    # Test case 4 - sustained, minimal time interval, no assessments available
    # Must yield an empty dataframe.
    test_case_4 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=40,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=-1,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=15,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_4.equals(
        test_dataframe[test_dataframe["timestamp"] > 50]
    ), "Test 4 failed!"
    # Test case 5 - time interval, assessments available
    # Must yield all assessments following the first one
    # until and including the assessment at 40.
    test_case_5 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=10,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=30,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=0,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_5.equals(test_dataframe.iloc[2:5]), "Test 5 failed!"
    # Test case 6 - time interval, no assessments available
    # Must yield an empty dataframe.
    test_case_6 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=10,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=50,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=0,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_6.equals(
        test_dataframe[test_dataframe["timestamp"] > 50]
    ), "Test 6 failed!"
    # Test case 7 - time interval, right side constrained
    # Next is further away, but within tolerance.
    test_case_7 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=10,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=5,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=10,
        opt_confirmation_time_right_side_max_tolerance=5,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_7.equals(test_dataframe.iloc[[2]]), "Test 7 failed!"
    # Test case 8 - time interval, right side constrained
    # Next is further away, and outside tolerance.
    test_case_8 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=10,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=5,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=10,
        opt_confirmation_time_right_side_max_tolerance=4,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_8.equals(
        test_dataframe[test_dataframe["timestamp"] > 50]
    ), "Test 8 failed!"
    # Test case 9 - time interval, left side tolerance standard
    # Must yield assesments at 20 and 30 (30 is first).
    test_case_9 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=10,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=15,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=0,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_9.equals(test_dataframe.iloc[2:4]), "Test 9 failed!"
    # Test case 10 - time interval, left side tolerance standard
    # Must yield assesment at 20, because with a tolerance of 5
    # days it is far away enough to be a confirmation score.
    test_case_10 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=10,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=15,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=0,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=5,
    )
    assert test_case_10.equals(test_dataframe.iloc[[2]]), "Test 10 failed!"
    # Test case 11 - last value only
    # Must yield the assessment at 40.
    test_case_11 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=10,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=30,
        opt_confirmation_included_values="last",
        opt_confirmation_sustained_minimal_distance=0,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_11.equals(test_dataframe.iloc[[4]]), "Test 11 failed!"
    # Test case 12 - time interval, left side tolerance standard, last only
    # Must yield assesment at 20, because with a tolerance of 5 days it is
    # far away enough to be a confirmation score. As the only assessment, it
    # is also the 'last' to be returned.
    test_case_12 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=10,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=15,
        opt_confirmation_included_values="last",
        opt_confirmation_sustained_minimal_distance=0,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=5,
    )
    assert test_case_12.equals(test_dataframe.iloc[[2]]), "Test 12 failed!"
    # Test case 13 - left side tolerance back to event, must not be its
    # own confirmation score
    # Must yield the assessment at 20.
    test_case_13 = edssprogression.EDSSProgression(
        time_column_name="timestamp",
    )._get_confirmation_scores_dataframe(
        current_timestamp=10,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=10,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=0,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=10,
    )
    assert test_case_13.equals(test_dataframe.iloc[[2]]), "Test 13 failed!"


def test_check_confirmation_scores_and_get_confirmed_score():
    # Note that the timestamps are not required anymore!
    # TODO: ADD TESTS FOR ADDITIONAL LOWER THRESHOLD!
    test_dataframe = pd.DataFrame({"score": [5, 4.5, 5]})
    # Test case 1 - minimum, against a reference of 4
    # Not confirmed, confirmed score is nan.
    assert edssprogression.EDSSProgression(
        opt_confirmation_type="minimum",
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
        edss_score_column_name="score",
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=5,
        current_reference=4,
        confirmation_scores_dataframe=test_dataframe,
        additional_lower_threshold=0,
    ) == (
        False,
        np.nan,
    ), "Test 1 failed!"
    # Test case 2 - minimum, against a reference of 3.5
    # Confirmed, confirmed score is 3.5.
    assert edssprogression.EDSSProgression(
        opt_confirmation_type="minimum",
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
        edss_score_column_name="score",
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=5,
        current_reference=3.5,
        confirmation_scores_dataframe=test_dataframe,
        additional_lower_threshold=0,
    ) == (
        True,
        4.5,
    ), "Test 2 failed!"
    # Test case 3 - monotonic, against a reference of 3.5
    # Not confirmed, confirmed score is nan
    assert edssprogression.EDSSProgression(
        opt_confirmation_type="monotonic",
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
        edss_score_column_name="score",
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=5,
        current_reference=3.5,
        confirmation_scores_dataframe=test_dataframe,
        additional_lower_threshold=0,
    ) == (
        False,
        np.nan,
    ), "Test 3 failed!"

    # Test case 4 - minimum, one score
    # Confirmed, confirmed score is 5.
    assert edssprogression.EDSSProgression(
        opt_confirmation_type="monotonic",
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
        edss_score_column_name="score",
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=5,
        current_reference=4.0,
        confirmation_scores_dataframe=pd.DataFrame({"score": [5]}),
        additional_lower_threshold=0,
    ) == (
        True,
        5,
    ), "Test 4 failed!"
    # Test case 5 - no scores available
    # Not confirmed, confirmed score is nan.
    assert edssprogression.EDSSProgression(
        opt_confirmation_type="monotonic",
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
        edss_score_column_name="score",
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=5,
        current_reference=4.0,
        confirmation_scores_dataframe=pd.DataFrame({"score": []}),
        additional_lower_threshold=0,
    ) == (
        False,
        np.nan,
    ), "Test 5 failed!"


def test_backtrack_minimal_distance_compatible_reference():
    test_dataframe = pd.DataFrame(
        {
            "baseline_timestamp": [0, 10, 20, 30, 40, 50],
            "baseline_score": [6.0, 5.5, 5.5, 5.0, 5.0, 4.5],
        }
    )
    # Test case 1 - minimal distance is larger than that to the
    # previous reference. Must yield the score 5.0 at time 40 for
    # the closest acceptable reference.
    assert edssprogression.EDSSProgression(
        opt_minimal_distance_time=20,
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
    )._backtrack_minimal_distance_compatible_reference(
        current_edss=6,
        current_timestamp=60,
        baselines_df=test_dataframe,
    ) == (
        5.0,
        40,
    ), "Test 1 failed!"
    # Test case 2 - minimal distance is larger than that to the
    # previous reference. No reference score far enough away is
    # low enough to serve as a progression reference.
    assert edssprogression.EDSSProgression(
        opt_minimal_distance_time=20,
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
    )._backtrack_minimal_distance_compatible_reference(
        current_edss=5.5,
        current_timestamp=60,
        baselines_df=test_dataframe,
    ) == (
        np.nan,
        np.nan,
    ), "Test 2 failed!"


# ----------------------------
# Part 2 - relapse-independent
# -----------------------------


def raw_pira_progression_result_is_equal_to_target(
    follow_up_dataframe,
    targets_dict,
    relapse_timestamps=[],
    ignore_relapses=False,
    args_dict={},
):
    annotated_df = edssprogression.EDSSProgression(
        **args_dict
    ).add_progression_events_to_follow_up(
        follow_up_dataframe=follow_up_dataframe,
        relapse_timestamps=relapse_timestamps,
    )

    # Initialize target dataframe
    target_df = follow_up_dataframe.copy()
    target_df["days_since_previous_relapse"] = np.nan
    target_df["days_to_next_relapse"] = np.nan
    target_df["is_post_event_rebaseline"] = False
    target_df["is_general_rebaseline"] = False
    target_df["is_raw_pira_rebaseline"] = False
    target_df["edss_score_used_as_new_general_reference"] = np.nan
    target_df["edss_score_used_as_new_raw_pira_reference"] = np.nan
    target_df["is_progression"] = False
    target_df["progression_type"] = None
    target_df["progression_score"] = np.nan
    target_df["progression_reference_score"] = np.nan
    target_df["progression_event_id"] = np.nan
    target_df["is_post_relapse_rebaseline"] = False

    # Remove some irrelevant relapse-related columns
    if ignore_relapses:
        annotated_df = annotated_df.drop(
            columns=[
                "days_since_previous_relapse",
                "days_to_next_relapse",
                "is_raw_pira_rebaseline",
                "is_post_relapse_rebaseline",
                "edss_score_used_as_new_raw_pira_reference",
            ]
        ).copy()
        target_df = target_df.drop(
            columns=[
                "days_since_previous_relapse",
                "days_to_next_relapse",
                "is_raw_pira_rebaseline",
                "is_post_relapse_rebaseline",
                "edss_score_used_as_new_raw_pira_reference",
            ]
        ).copy()

    target_df = target_df.set_index("days_after_baseline")
    for target_column in targets_dict:
        for target in targets_dict.get(target_column, []):
            target_df.at[target[0], target_column] = target[1]
    target_df = target_df.reset_index()

    return annotated_df.equals(target_df)


def test_relapse_independent_confirmation():
    # Unconfirmed vs. next-confirmed vs. sustained
    test_dataframe_no_next_sustained = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50],
            "edss_score": [1, 1, 1.5, 2.0, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_no_next_sustained,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 2.0)],
            "progression_reference_score": [(30, 1.0)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
        },
    ), "Test 1 'unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_no_next_sustained,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 2.0)],
            "progression_reference_score": [(30, 1.0)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 2 'next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_no_next_sustained,
        targets_dict={},
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
        },
    ), "Test 3 'sustained' failed!"

    # Test various confirmation durations
    test_dataframe_durations = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40],
            "edss_score": [1, 2.5, 2.5, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_durations,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.5)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.5)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 4 '10 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_durations,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.0)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 5 '20 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_durations,
        targets_dict={},
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_baseline_type": "fixed",
        },
    ), "Test 6 '30 units confirmed' failed!"

    # Test left-hand side tolerance
    test_dataframe_left_tolerance = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40],
            "edss_score": [1, 2.5, 2.5, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_left_tolerance,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.0)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 0,
            "opt_baseline_type": "fixed",
        },
    ), "Test 7 '15 units confirmed, no tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_left_tolerance,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.5)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.5)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 8 '15 units confirmed, 5 units tolerance' failed!"

    # Test right-hand side max. distance constraint
    test_dataframe_right_constraint = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 30, 40],
            "edss_score": [1, 2.5, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_right_constraint,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.0)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_time_right_side_max_tolerance": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 9 '10 units confirmed, 10 units tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_right_constraint,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 2.0)],
            "progression_reference_score": [(30, 1.0)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_time_right_side_max_tolerance": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 10 '10 units confirmed, 5 units tolerance' failed!"

    # Test minimal distance for sustained
    test_dataframe_sustained_minimal_distance = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30],
            "edss_score": [1, 2.5, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_sustained_minimal_distance,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.0)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 11 'Sustained, minimum 20 units' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_sustained_minimal_distance,
        targets_dict={},
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 21,
            "opt_baseline_type": "fixed",
        },
    ), "Test 12 'Sustained, minimum 21 units' failed!"

    # Test all vs. last confirmed
    test_dataframe_all_vs_last = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40],
            "edss_score": [1, 2.0, 1.5, 1.5, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_all_vs_last,
        targets_dict={},
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "opt_baseline_type": "fixed",
        },
    ), "Test 13 '30 units confirmed, all values' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_all_vs_last,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.0)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "last",
            "opt_baseline_type": "fixed",
        },
    ), "Test 14 '30 units confirmed, last only' failed!"

    # Minimum vs. monotonic
    test_dataframe_min_vs_monotonic = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40],
            "edss_score": [1, 2.5, 2.0, 2.0, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_min_vs_monotonic,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.0)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_type": "minimum",
            "opt_baseline_type": "fixed",
        },
    ), "Test 15 '30 units confirmed, minimum' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_min_vs_monotonic,
        targets_dict={},
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_type": "monotonic",
            "opt_baseline_type": "fixed",
        },
    ), "Test 16 '30 units confirmed, monotonic' failed!"

    # Minimum/monotonic - correct event scores?
    test_dataframe_min_vs_monotonic_event_scores = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40],
            "edss_score": [1, 2.5, 2.5, 3.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_min_vs_monotonic_event_scores,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.5)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.5)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_type": "minimum",
            "opt_baseline_type": "fixed",
        },
    ), "Test 17 '20 units confirmed, minimum' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_min_vs_monotonic_event_scores,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.5)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.5)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_type": "monotonic",
            "opt_baseline_type": "fixed",
        },
    ), "Test 18 '20 units confirmed, monotonic' failed!"

    # No confirmation requirement for last assessment
    test_dataframe_last_confirmed = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_last_confirmed,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 2)],
            "progression_reference_score": [(30, 1.0)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": True,
        },
    ), "Test 19 'Last requires confirmation' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_last_confirmed,
        targets_dict={
            "is_post_event_rebaseline": [(30, True), (50, True)],
            "is_general_rebaseline": [(30, True), (50, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0), (50, 3.0)],
            "is_progression": [(30, True), (50, True)],
            "progression_type": [(30, LABEL_PIRA), (50, LABEL_PIRA)],
            "progression_score": [(30, 2), (50, 3)],
            "progression_reference_score": [(30, 1.0), (50, 2.0)],
            "progression_event_id": [(30, 1), (50, 2)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": False,
        },
    ), "Test 20 'Last does not require confirmation' failed!"


def test_relapse_independent_baselines():
    # Fixed vs. roving without/with confirmation
    test_dataframe_fixed_roving = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60],
            "edss_score": [5.0, 4.0, 4.5, 4.0, 4.0, 4.5, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_fixed_roving,
        targets_dict={},
        args_dict={
            "opt_baseline_type": "fixed",
        },
    ), "Test 1 'Fixed baseline' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_fixed_roving,
        targets_dict={
            "is_general_rebaseline": [(10, True), (60, True)],
            "edss_score_used_as_new_general_reference": [(10, 4.0), (60, 3.5)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_roving_reference_confirmation_time": 0,
        },
    ), "Test 2 'Roving unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_fixed_roving,
        targets_dict={
            "is_general_rebaseline": [(10, True), (30, True)],
            "edss_score_used_as_new_general_reference": [(10, 4.5), (30, 4.0)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
        },
    ), "Test 3 'Roving next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_fixed_roving,
        targets_dict={
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 4.5)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
        },
    ), "Test 4 'Roving 20 units confirmed' failed!"

    # Roving reference all vs. last confirmed
    test_dataframe_roving_all_vs_last = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60],
            "edss_score": [5.0, 4.0, 4.5, 4.0, 4.0, 4.5, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_roving_all_vs_last,
        targets_dict={
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 4.5)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
            "opt_roving_reference_confirmation_included_values": "all",
        },
    ), "Test 5 'Roving reference, 20 units confirmed, all values' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_roving_all_vs_last,
        targets_dict={
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 4.0)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
            "opt_roving_reference_confirmation_included_values": "last",
        },
    ), "Test 6 'Roving reference, 20 units confirmed, all values' failed!"

    # Roving reference with left- or right-hand tolerance/constraint
    test_dataframe_roving_left_right = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60],
            "edss_score": [5.0, 4.0, 4.0, 4.5, 4.0, 4.5, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_roving_left_right,
        targets_dict={
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 4.5)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 15,
            "opt_roving_reference_confirmation_time_left_side_max_tolerance": 0,
            "opt_roving_reference_confirmation_time_right_side_max_tolerance": np.inf,
        },
    ), "Test 7 'Roving reference, 15 units confirmed, no left hand side tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_roving_left_right,
        targets_dict={
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 4.0)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 15,
            "opt_roving_reference_confirmation_time_left_side_max_tolerance": 5,
            "opt_roving_reference_confirmation_time_right_side_max_tolerance": np.inf,
        },
    ), "Test 8 'Roving reference, 15 units confirmed, 5 units left hand side tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_roving_left_right,
        targets_dict={
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 4.0)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 5,
            "opt_roving_reference_confirmation_time_left_side_max_tolerance": 0,
            "opt_roving_reference_confirmation_time_right_side_max_tolerance": 5,
        },
    ), "Test 9 'Roving reference, 5 units confirmed, no right hand side constraint' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_roving_left_right,
        targets_dict={},
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 5,
            "opt_roving_reference_confirmation_time_left_side_max_tolerance": 0,
            "opt_roving_reference_confirmation_time_right_side_max_tolerance": 4,
        },
    ), "Test 10 'Roving reference, 5 units confirmed, 4 units right hand side constraint' failed!"


def test_roving_baseline_with_minimal_score():
    # Fixed vs. roving without/with confirmation
    test_dataframe_roving_lowest_1 = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40],
            "edss_score": [5.0, 4.0, 3.5, 4.0, 4.0],
        }
    )
    # All confirmed with subsequent drop and increase
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_roving_lowest_1,
        targets_dict={
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 3.5)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
            "opt_roving_reference_confirmation_included_values": "all",
            "opt_roving_reference_use_lowest_confirmation_score": True,
        },
    ), "Test 1 for roving with minimal score failed!"
    # Last confirmed with subsequent drop and increase
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_roving_lowest_1,
        targets_dict={
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 4.0)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
            "opt_roving_reference_confirmation_included_values": "last",
            "opt_roving_reference_use_lowest_confirmation_score": True,
        },
    ), "Test 2 for roving with minimal score failed!"
    test_dataframe_roving_lowest_2 = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40],
            "edss_score": [5.0, 4.0, 4.0, 3.5, 4.0],
        }
    )
    # All confirmed with repetition and drop
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_roving_lowest_2,
        targets_dict={
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 3.5)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
            "opt_roving_reference_confirmation_included_values": "all",
            "opt_roving_reference_use_lowest_confirmation_score": True,
        },
    ), "Test 3 for roving with minimal score failed!"
    # Last confirmed with repetition and drop
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_roving_lowest_2,
        targets_dict={
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 3.5)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
            "opt_roving_reference_confirmation_included_values": "last",
            "opt_roving_reference_use_lowest_confirmation_score": True,
        },
    ), "Test 4 for roving with minimal score failed!"
    # Inconsistency part 1
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=pd.DataFrame(
            {
                "days_after_baseline": [0, 10, 20, 30, 40],
                "edss_score": [5.0, 4.0, 3.5, 4.5, 5.0],
            }
        ),
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(10, True), (30, True)],
            "edss_score_used_as_new_general_reference": [(10, 3.5), (30, 4.5)],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 4.5)],
            "progression_reference_score": [(30, 3.5)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 10,
            "opt_roving_reference_confirmation_included_values": "all",
            "opt_roving_reference_use_lowest_confirmation_score": True,
        },
    ), "Test 5 for roving with minimal score failed!"
    # Inconsistency part 2
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=pd.DataFrame(
            {
                "days_after_baseline": [0, 10, 20, 30, 40],
                "edss_score": [5.0, 3.5, 4.0, 4.5, 5.0],
            }
        ),
        targets_dict={},
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 10,
            "opt_roving_reference_confirmation_included_values": "all",
            "opt_roving_reference_use_lowest_confirmation_score": True,
        },
    ), "Test 6 for roving with minimal score failed!"
    # Inconsistency part 3
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=pd.DataFrame(
            {
                "days_after_baseline": [0, 10, 20, 30, 40],
                "edss_score": [5.0, 4.0, 4.0, 4.5, 5.0],
            }
        ),
        targets_dict={
            "is_post_event_rebaseline": [(40, True)],
            "is_general_rebaseline": [(10, True), (40, True)],
            "edss_score_used_as_new_general_reference": [(10, 4.0), (40, 5.0)],
            "is_progression": [(40, True)],
            "progression_type": [(40, LABEL_PIRA)],
            "progression_score": [(40, 5.0)],
            "progression_reference_score": [(40, 4.0)],
            "progression_event_id": [(40, 1)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 10,
            "opt_roving_reference_confirmation_included_values": "all",
            "opt_roving_reference_use_lowest_confirmation_score": True,
        },
    ), "Test 5 for roving with minimal score failed!"


def test_relapse_independent_minimal_distance():
    # Minimal distance to reference, various distances
    test_dataframe_distances_to_reference = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30],
            "edss_score": [1, 2.0, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_distances_to_reference,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.0)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 10,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 1 '10 units to reference' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_distances_to_reference,
        targets_dict={
            "is_post_event_rebaseline": [(20, True)],
            "is_general_rebaseline": [(20, True)],
            "edss_score_used_as_new_general_reference": [(20, 2.0)],
            "is_progression": [(20, True)],
            "progression_type": [(20, LABEL_PIRA)],
            "progression_score": [(20, 2.0)],
            "progression_reference_score": [(20, 1.0)],
            "progression_event_id": [(20, 1)],
        },
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 20,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 2 '20 units to reference' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_distances_to_reference,
        targets_dict={},
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 31,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 3 '31 units to reference' failed!"

    # Minimal distance to reference with confirmation
    test_dataframe_distances_to_reference_confirmed = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30],
            "edss_score": [1, 2.0, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_distances_to_reference_confirmed,
        targets_dict={
            "is_post_event_rebaseline": [(20, True)],
            "is_general_rebaseline": [(20, True)],
            "edss_score_used_as_new_general_reference": [(20, 2.0)],
            "is_progression": [(20, True)],
            "progression_type": [(20, LABEL_PIRA)],
            "progression_score": [(20, 2.0)],
            "progression_reference_score": [(20, 1.0)],
            "progression_event_id": [(20, 1)],
        },
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 20,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
        },
    ), "Test 4 '20 units to reference, unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_distances_to_reference_confirmed,
        targets_dict={
            "is_post_event_rebaseline": [(20, True)],
            "is_general_rebaseline": [(20, True)],
            "edss_score_used_as_new_general_reference": [(20, 2.0)],
            "is_progression": [(20, True)],
            "progression_type": [(20, LABEL_PIRA)],
            "progression_score": [(20, 2.0)],
            "progression_reference_score": [(20, 1.0)],
            "progression_event_id": [(20, 1)],
        },
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 20,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
        },
    ), "Test 5 '20 units to reference, 10 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_distances_to_reference_confirmed,
        targets_dict={},
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 20,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
        },
    ), "Test 6 '20 units to reference, 15 units confirmed' failed!"

    # Minimal distance to previous
    test_dataframe_distances_to_previous = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30],
            "edss_score": [1, 2.0, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_distances_to_previous,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_progression": [(10, True)],
            "progression_type": [(10, LABEL_PIRA)],
            "progression_score": [(10, 2.0)],
            "progression_reference_score": [(10, 1.0)],
            "progression_event_id": [(10, 1)],
        },
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 10,
            "opt_minimal_distance_type": "previous",
        },
    ), "Test 7 '10 units to previous' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_distances_to_previous,
        targets_dict={},
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 11,
            "opt_minimal_distance_type": "previous",
        },
    ), "Test 8 '11 units to previous' failed!"

    # Backtracking
    test_dataframe_backtracking = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30],
            "edss_score": [3.5, 3.0, 2.5, 4.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_backtracking,
        targets_dict={
            "is_general_rebaseline": [(10, True), (20, True)],
            "edss_score_used_as_new_general_reference": [(10, 3.0), (20, 2.5)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": False,
        },
    ), "Test 9 '15 units to reference, without backtracking' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_backtracking,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(10, True), (20, True), (30, True)],
            "edss_score_used_as_new_general_reference": [
                (10, 3.0),
                (20, 2.5),
                (30, 4.0),
            ],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 4.0)],
            "progression_reference_score": [(30, 3.0)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 10 '15 units to reference, with backtracking' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_backtracking,
        targets_dict={
            "is_general_rebaseline": [(10, True), (20, True)],
            "edss_score_used_as_new_general_reference": [(10, 3.0), (20, 2.5)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 25,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 10 '25 units to reference, with backtracking' failed!"

    # Backtracking with confirmation
    test_dataframe_backtracking_confirmed = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60],
            "edss_score": [3.5, 3.0, 2.5, 4.5, 4.5, 4.0, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_backtracking_confirmed,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [
                (10, True),
                (20, True),
                (30, True),
                (50, True),
                (60, True),
            ],
            "edss_score_used_as_new_general_reference": [
                (10, 3.0),
                (20, 2.5),
                (30, 4.5),
                (50, 4.0),
                (60, 3.5),
            ],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 4.5)],
            "progression_reference_score": [(30, 3.0)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
        },
    ), "Test 11 '15 units to reference, with backtracking, 10 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_backtracking_confirmed,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [
                (10, True),
                (20, True),
                (30, True),
                (60, True),
            ],
            "edss_score_used_as_new_general_reference": [
                (10, 3.0),
                (20, 2.5),
                (30, 4.0),
                (60, 3.5),
            ],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 4.0)],
            "progression_reference_score": [(30, 3.0)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
        },
    ), "Test 12 '15 units to reference, with backtracking, 20 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_backtracking_confirmed,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(10, True), (20, True), (30, True)],
            "edss_score_used_as_new_general_reference": [
                (10, 3.0),
                (20, 2.5),
                (30, 3.5),
            ],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 3.5)],
            "progression_reference_score": [(30, 2.5)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 0,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
        },
    ), "Test 13 'No minimal distance, 30 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_backtracking_confirmed,
        targets_dict={
            "is_general_rebaseline": [(10, True), (20, True)],
            "edss_score_used_as_new_general_reference": [(10, 3.0), (20, 2.5)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
        },
    ), "Test 14 '15 units to reference, with backtracking, 30 units confirmed' failed!"


def test_relapse_independent_first_vs_all_events():
    test_dataframe_first_all_events = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 70],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_first_all_events,
        targets_dict={
            "is_post_event_rebaseline": [(30, True), (50, True), (70, True)],
            "is_general_rebaseline": [(30, True), (50, True), (70, True)],
            "edss_score_used_as_new_general_reference": [
                (30, 2.0),
                (50, 3.0),
                (70, 4.0),
            ],
            "is_progression": [(30, True), (50, True), (70, True)],
            "progression_type": [(30, LABEL_PIRA), (50, LABEL_PIRA), (70, LABEL_PIRA)],
            "progression_score": [(30, 2.0), (50, 3.0), (70, 4.0)],
            "progression_reference_score": [(30, 1.0), (50, 2.0), (70, 3.0)],
            "progression_event_id": [(30, 1), (50, 2), (70, 3)],
        },
        args_dict={
            "return_first_event_only": False,
        },
    ), "Test 1 'Return all events' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_first_all_events,
        targets_dict={
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 2.0)],
            "progression_reference_score": [(30, 1.0)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "return_first_event_only": True,
        },
    ), "Test 2 'Return first event only' failed!"


def test_relapse_independent_multiple_events_rebaselining():
    # Fixed vs. roving
    test_dataframe_multiple_fixed_roving = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 70],
            "edss_score": [1, 1, 1.5, 2.0, 2.0, 1.5, 1.5, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_multiple_fixed_roving,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 2.0)],
            "progression_reference_score": [(30, 1.0)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_roving_reference_require_confirmation": False,
        },
    ), "Test 1 'Fixed baseline' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_multiple_fixed_roving,
        targets_dict={
            "is_post_event_rebaseline": [(30, True), (70, True)],
            "is_general_rebaseline": [(30, True), (50, True), (70, True)],
            "edss_score_used_as_new_general_reference": [
                (30, 2.0),
                (50, 1.5),
                (70, 2.5),
            ],
            "is_progression": [(30, True), (70, True)],
            "progression_type": [(30, LABEL_PIRA), (70, LABEL_PIRA)],
            "progression_score": [(30, 2.0), (70, 2.5)],
            "progression_reference_score": [(30, 1.0), (70, 1.5)],
            "progression_event_id": [(30, 1), (70, 2)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
        },
    ), "Test 2 'Roving reference' failed!"

    # With and without confirmation
    test_dataframe_multiple_confirmation = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "edss_score": [1, 1, 1.5, 2.5, 2.0, 1.5, 1.5, 2.5, 3.0, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_multiple_confirmation,
        targets_dict={
            "is_post_event_rebaseline": [(30, True), (90, True)],
            "is_general_rebaseline": [(30, True), (90, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.5), (90, 3.5)],
            "is_progression": [(30, True), (90, True)],
            "progression_type": [(30, LABEL_PIRA), (90, LABEL_PIRA)],
            "progression_score": [(30, 2.5), (90, 3.5)],
            "progression_reference_score": [(30, 1.0), (90, 2.5)],
            "progression_event_id": [(30, 1), (90, 2)],
        },
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
        },
    ), "Test 3 'Fixed baseline, progression unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=True,
        follow_up_dataframe=test_dataframe_multiple_confirmation,
        targets_dict={
            "is_post_event_rebaseline": [(30, True), (80, True)],
            "is_general_rebaseline": [(30, True), (80, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0), (80, 3.0)],
            "is_progression": [(30, True), (80, True)],
            "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA)],
            "progression_score": [(30, 2.0), (80, 3.0)],
            "progression_reference_score": [(30, 1.0), (80, 2.0)],
            "progression_event_id": [(30, 1), (80, 2)],
        },
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 4 'Fixed baseline, progression next-confirmed' failed!"


# ----------------------
# Part 3 - with relapses
# ----------------------


def test_add_relapses_to_follow_up():
    test_dataframe = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 70],
            "edss_score": [1, 1, 1.5, 2.0, 2.0, 1.5, 1.5, 2.5],
        }
    )

    # Test case 1 - no relapses, must yield empty dataframe
    target_case_1 = test_dataframe.copy()
    target_case_1["days_since_previous_relapse"] = np.nan
    target_case_1["days_to_next_relapse"] = np.nan
    assert (
        edssprogression.EDSSProgression()
        ._add_relapses_to_follow_up(
            follow_up_df=test_dataframe,
            relapse_timestamps=[],
        )
        .equals(target_case_1)
    ), "Test 1 'No relapses' failed!"

    # Test case 2 - one relapse at assessment (at 20)
    target_case_2 = test_dataframe.copy()
    target_case_2 = pd.concat(
        [
            test_dataframe,
            pd.DataFrame(
                {
                    "days_since_previous_relapse": [
                        np.nan,
                        np.nan,
                        0,
                        10,
                        20,
                        30,
                        40,
                        50,
                    ],
                    "days_to_next_relapse": [
                        20,
                        10,
                        0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                }
            ),
        ],
        axis=1,
    )
    assert (
        edssprogression.EDSSProgression()
        ._add_relapses_to_follow_up(
            follow_up_df=test_dataframe,
            relapse_timestamps=[20],
        )
        .equals(target_case_2)
    ), "Test 2 'Relapse at 20' failed!"

    # Test case 3 - one relapse between assessments (at 25)
    target_case_3 = test_dataframe.copy()
    target_case_3 = pd.concat(
        [
            test_dataframe,
            pd.DataFrame(
                {
                    "days_since_previous_relapse": [
                        np.nan,
                        np.nan,
                        np.nan,
                        5,
                        15,
                        25,
                        35,
                        45,
                    ],
                    "days_to_next_relapse": [
                        25,
                        15,
                        5,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                }
            ),
        ],
        axis=1,
    )
    assert (
        edssprogression.EDSSProgression()
        ._add_relapses_to_follow_up(
            follow_up_df=test_dataframe,
            relapse_timestamps=[25],
        )
        .equals(target_case_3)
    ), "Test 3 'Relapse at 25' failed!"

    # Test case 4 - two relapses between two assessments (2 and 8)
    target_case_4 = test_dataframe.copy()
    target_case_4 = pd.concat(
        [
            test_dataframe,
            pd.DataFrame(
                {
                    "days_since_previous_relapse": [
                        np.nan,
                        2,
                        12,
                        22,
                        32,
                        42,
                        52,
                        62,
                    ],
                    "days_to_next_relapse": [
                        2,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                }
            ),
        ],
        axis=1,
    )
    assert (
        edssprogression.EDSSProgression()
        ._add_relapses_to_follow_up(
            follow_up_df=test_dataframe,
            relapse_timestamps=[2, 8],
        )
        .equals(target_case_4)
    ), "Test 4 'Relapses at 2 and 8' failed!"

    # Test case 5 - relapses before and after the follow-up (-5 and 80)
    target_case_5 = test_dataframe.copy()
    target_case_5 = pd.concat(
        [
            test_dataframe,
            pd.DataFrame(
                {
                    "days_since_previous_relapse": [
                        5,
                        15,
                        25,
                        35,
                        45,
                        55,
                        65,
                        75,
                    ],
                    "days_to_next_relapse": [
                        80,
                        70,
                        60,
                        50,
                        40,
                        30,
                        20,
                        10,
                    ],
                }
            ),
        ],
        axis=1,
    )
    assert (
        edssprogression.EDSSProgression()
        ._add_relapses_to_follow_up(
            follow_up_df=test_dataframe,
            relapse_timestamps=[-5, 80],
        )
        .equals(target_case_5)
    ), "Test 5 'Relapses at -5 and 80' failed!"


def test_get_post_relapse_rebaseline_timestamps():
    test_relapses_cases_1_2_3_4 = [6, 30]
    test_dataframe_cases_1_2_3_4 = pd.DataFrame(
        {"days_after_baseline": [0, 8, 36, 48, 60]}
    )
    # Test case 1 - assessments well-separated
    test_case_1_target = [8, 36]
    assert Counter(
        edssprogression.EDSSProgression(
            opt_raw_before_relapse_max_days=2,
            opt_raw_after_relapse_max_days=1,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_1_2_3_4,
            relapse_timestamps=test_relapses_cases_1_2_3_4,
        )
    ) == Counter(test_case_1_target), "Test 1 'Well-separated re-baselining' failed!"

    # Test case 2 - rebaseline of first relapse after second
    # relapse, within buffer
    test_case_2_target = [36, 48]
    assert Counter(
        edssprogression.EDSSProgression(
            opt_raw_before_relapse_max_days=4,
            opt_raw_after_relapse_max_days=12,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_1_2_3_4,
            relapse_timestamps=test_relapses_cases_1_2_3_4,
        )
    ) == Counter(
        test_case_2_target
    ), "Test 2 'Rebaseline of first relapse after second relapse, within buffer' failed!"

    # Test case 3 - rebaseline of first relapse after second relapse,
    # after second buffer (same for both)
    test_case_3_target = [36]
    assert Counter(
        edssprogression.EDSSProgression(
            opt_raw_before_relapse_max_days=4,
            opt_raw_after_relapse_max_days=4,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_1_2_3_4,
            relapse_timestamps=test_relapses_cases_1_2_3_4,
        )
    ) == Counter(
        test_case_3_target
    ), "Test 3 'Rebaseline of first relapse after second relapse, after second buffer' failed!"

    # Test case 4 - overlapping RAW windows
    test_case_4_target = [60]
    assert Counter(
        edssprogression.EDSSProgression(
            opt_raw_before_relapse_max_days=4,
            opt_raw_after_relapse_max_days=20,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_1_2_3_4,
            relapse_timestamps=test_relapses_cases_1_2_3_4,
        )
    ) == Counter(test_case_4_target), "Test 4 'Overlapping RAW windows failed!"

    # Test case 5 - rebaseline of first relapse before second, but within RAW window
    test_relapses_cases_5 = [6, 30]
    test_dataframe_cases_5 = pd.DataFrame({"days_after_baseline": [0, 8, 28, 48, 60]})
    test_case_5_target = [28, 48]
    assert Counter(
        edssprogression.EDSSProgression(
            opt_raw_before_relapse_max_days=4,
            opt_raw_after_relapse_max_days=4,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_5,
            relapse_timestamps=test_relapses_cases_5,
        )
    ) == Counter(
        test_case_5_target
    ), "Test 5 'Rebaseline within buffer of next' failed!"

    # Test case 6 - multiple non-overlapping and overlapping relapses
    test_relapses_cases_6 = [15, 25, 40, 48]
    test_dataframe_cases_6 = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 50, 60, 70],
        }
    )
    test_case_6_target = [50, 60]
    assert Counter(
        edssprogression.EDSSProgression(
            opt_raw_before_relapse_max_days=2,
            opt_raw_after_relapse_max_days=7,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_6,
            relapse_timestamps=test_relapses_cases_6,
        )
    ) == Counter(
        test_case_6_target
    ), "Test 6 'Rebaseline with multiple overlapping and non-overlapping' failed!"


def test_roving_raw_pira_descends_to_general():
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=False,
        relapse_timestamps=[15],
        follow_up_dataframe=pd.DataFrame(
            {
                "days_after_baseline": [0, 10, 20, 30, 40, 50, 60],
                "edss_score": [2.0, 1.0, 3.5, 3.0, 1.0, 2.0, 3.5],
            }
        ),
        targets_dict={
            "days_to_next_relapse": [(0, 15), (10, 5)],
            "days_since_previous_relapse": [
                (20, 5),
                (30, 15),
                (40, 25),
                (50, 35),
                (60, 45),
            ],
            "is_post_event_rebaseline": [],
            "is_general_rebaseline": [],
            "edss_score_used_as_new_general_reference": [],
            "is_raw_pira_rebaseline": [(20, True), (30, True), (40, True)],
            "edss_score_used_as_new_raw_pira_reference": [
                (20, 3.5),
                (30, 3.0),
                (40, 2.0),
            ],
            "is_post_relapse_rebaseline": [(20, True)],
            "is_progression": [],
            "progression_type": [],
            "progression_score": [],
            "progression_event_id": [],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_raw_after_relapse_max_days": 3,
            "opt_raw_before_relapse_max_days": 3,
        },
    ), "Roving baseline for RAW/PIRA is bugged..."


def test_raw_pira_unconfirmed():
    test_dataframe = pd.DataFrame(
        {"days_after_baseline": [0, 20, 40], "edss_score": [1, 2.0, 2.0]}
    )
    # Test case 1 - no relapses
    test_case_1_targets = {
        "days_since_previous_relapse": [],
        "days_to_next_relapse": [],
        "is_post_event_rebaseline": [(20, True)],
        "is_general_rebaseline": [(20, True)],
        "is_raw_pira_rebaseline": [(20, True)],
        "edss_score_used_as_new_general_reference": [(20, 2.0)],
        "edss_score_used_as_new_raw_pira_reference": [(20, 2.0)],
        "is_progression": [(20, True)],
        "progression_type": [(20, LABEL_PIRA)],
        "progression_score": [(20, 2.0)],
        "progression_reference_score": [(20, 1.0)],
        "progression_event_id": [(20, 1)],
        "is_post_relapse_rebaseline": [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[],
        targets_dict=test_case_1_targets,
        args_dict={},
    ), "Test 1 failed!"

    # Test cases 2 and 3 - relapse shortly before progression
    # The goal of these cases is to test the correct comparator sign (> vs. >=)
    test_case_2_targets = {
        "days_since_previous_relapse": [(20, 5), (40, 25)],
        "days_to_next_relapse": [(0, 15)],
        "is_post_event_rebaseline": [(20, True)],
        "is_general_rebaseline": [(20, True)],
        "is_raw_pira_rebaseline": [(20, True)],
        "edss_score_used_as_new_general_reference": [(20, 2.0)],
        "edss_score_used_as_new_raw_pira_reference": [(20, 2.0)],
        "is_progression": [(20, True)],
        "progression_type": [(20, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(20, 2.0)],
        "progression_reference_score": [(20, 1.0)],
        "progression_event_id": [(20, 1)],
        "is_post_relapse_rebaseline": [(20, True)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict=test_case_2_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 4,
        },
    ), "Test 2 failed!"
    test_case_3_targets = {
        "days_since_previous_relapse": [(20, 5), (40, 25)],
        "days_to_next_relapse": [(0, 15)],
        "is_post_event_rebaseline": [(20, True)],
        "is_general_rebaseline": [(20, True)],
        "is_raw_pira_rebaseline": [(20, True)],
        "edss_score_used_as_new_general_reference": [(20, 2.0)],
        "edss_score_used_as_new_raw_pira_reference": [(20, 2.0)],
        "is_progression": [(20, True)],
        "progression_type": [(20, LABEL_RAW)],
        "progression_score": [(20, 2.0)],
        "progression_reference_score": [(20, 1.0)],
        "progression_event_id": [(20, 1)],
        "is_post_relapse_rebaseline": [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict=test_case_3_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 5,
        },
    ), "Test 3 failed!"

    # Test cases 4 and 5 - relapse shortly after progression
    # The goal of these cases is to test the correct comparator sign (< vs. <=)
    test_case_4_targets = {
        "days_since_previous_relapse": [(40, 15)],
        "days_to_next_relapse": [(0, 25), (20, 5)],
        "is_post_event_rebaseline": [(20, True)],
        "is_general_rebaseline": [(20, True)],
        "is_raw_pira_rebaseline": [(20, True)],
        "edss_score_used_as_new_general_reference": [(20, 2.0)],
        "edss_score_used_as_new_raw_pira_reference": [(20, 2.0)],
        "is_progression": [(20, True)],
        "progression_type": [(20, LABEL_PIRA)],
        "progression_score": [(20, 2.0)],
        "progression_reference_score": [(20, 1.0)],
        "progression_event_id": [(20, 1)],
        "is_post_relapse_rebaseline": [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[25],
        targets_dict=test_case_4_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 4,
            "opt_raw_after_relapse_max_days": 5,
        },
    ), "Test 4 failed!"
    test_case_5_targets = {
        "days_since_previous_relapse": [(40, 15)],
        "days_to_next_relapse": [(0, 25), (20, 5)],
        "is_post_event_rebaseline": [(20, True)],
        "is_general_rebaseline": [(20, True)],
        "is_raw_pira_rebaseline": [(20, True)],
        "edss_score_used_as_new_general_reference": [(20, 2.0)],
        "edss_score_used_as_new_raw_pira_reference": [(20, 2.0)],
        "is_progression": [(20, True)],
        "progression_type": [(20, LABEL_RAW)],
        "progression_score": [(20, 2.0)],
        "progression_reference_score": [(20, 1.0)],
        "progression_event_id": [(20, 1)],
        "is_post_relapse_rebaseline": [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[25],
        targets_dict=test_case_5_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 5,
        },
    ), "Test 5 failed!"


def test_raw_pira_confirmed():
    test_dataframe = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 70, 80],
            "edss_score": [1, 1, 1.5, 2.0, 2.0, 3.0, 2.5, 2.5, 3.0],
        }
    )
    test_relapse_timestamps = [45]

    # Test case 1 - no relapses
    test_case_1_targets = {
        "days_since_previous_relapse": [(50, 5), (60, 15), (70, 25), (80, 35)],
        "days_to_next_relapse": [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        "is_post_event_rebaseline": [(30, True), (50, True)],
        "is_general_rebaseline": [(30, True), (50, True), (60, True)],
        "is_raw_pira_rebaseline": [(30, True), (50, True), (60, True)],
        "edss_score_used_as_new_general_reference": [(30, 2.0), (50, 3.0), (60, 2.5)],
        "edss_score_used_as_new_raw_pira_reference": [(30, 2.0), (50, 3.0), (60, 2.5)],
        "is_progression": [(30, True), (50, True)],
        "progression_type": [(30, LABEL_PIRA), (50, LABEL_RAW)],
        "progression_score": [(30, 2.0), (50, 3.0)],
        "progression_reference_score": [(30, 1.0), (50, 2.0)],
        "progression_event_id": [(30, 1), (50, 2)],
        "is_post_relapse_rebaseline": [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_1_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
            "opt_confirmation_included_values": "all",
            "opt_pira_allow_relapses_between_event_and_confirmation": False,
        },
    ), "Test 1 failed!"

    # Test case 2 - confirmation score in RAW window
    test_case_2_targets = {
        "days_since_previous_relapse": [(50, 5), (60, 15), (70, 25), (80, 35)],
        "days_to_next_relapse": [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        "is_post_event_rebaseline": [(30, True)],
        "is_general_rebaseline": [(30, True)],
        "is_raw_pira_rebaseline": [(30, True), (60, True)],
        "edss_score_used_as_new_general_reference": [(30, 2.0)],
        "edss_score_used_as_new_raw_pira_reference": [(30, 2.0), (60, 2.5)],
        "is_progression": [(30, True)],
        "progression_type": [(30, LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW)],
        "progression_score": [(30, 2.0)],
        "progression_reference_score": [(30, 1.0)],
        "progression_event_id": [(30, 1)],
        "is_post_relapse_rebaseline": [(60, True)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_2_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_included_values": "all",
            "opt_pira_allow_relapses_between_event_and_confirmation": False,
        },
    ), "Test 2 failed!"

    # Test case 3 - confirmation score before RAW window
    test_case_3_targets = {
        "days_since_previous_relapse": [(50, 5), (60, 15), (70, 25), (80, 35)],
        "days_to_next_relapse": [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        "is_post_event_rebaseline": [(30, True)],
        "is_general_rebaseline": [(30, True)],
        "is_raw_pira_rebaseline": [(30, True), (60, True)],
        "edss_score_used_as_new_general_reference": [(30, 2.0)],
        "edss_score_used_as_new_raw_pira_reference": [(30, 2.0), (60, 2.5)],
        "is_progression": [(30, True)],
        "progression_type": [(30, LABEL_PIRA)],
        "progression_score": [(30, 2.0)],
        "progression_reference_score": [(30, 1.0)],
        "progression_event_id": [(30, 1)],
        "is_post_relapse_rebaseline": [(60, True)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_3_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 4,
            "opt_raw_after_relapse_max_days": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_included_values": "all",
            "opt_pira_allow_relapses_between_event_and_confirmation": False,
        },
    ), "Test 3 failed!"
    # Test case 4 - (2, 10, True, 30, "all", True, "Relapse within confirmation interval")
    test_case_4_targets = {
        "days_since_previous_relapse": [(50, 5), (60, 15), (70, 25), (80, 35)],
        "days_to_next_relapse": [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        "is_post_event_rebaseline": [(30, True)],
        "is_general_rebaseline": [(30, True)],
        "is_raw_pira_rebaseline": [(30, True), (60, True)],
        "edss_score_used_as_new_general_reference": [(30, 2.0)],
        "edss_score_used_as_new_raw_pira_reference": [(30, 2.0), (60, 2.5)],
        "is_progression": [(30, True)],
        "progression_type": [(30, LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW)],
        "progression_score": [(30, 2.0)],
        "progression_reference_score": [(30, 1.0)],
        "progression_event_id": [(30, 1)],
        "is_post_relapse_rebaseline": [(60, True)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_4_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 2,
            "opt_raw_after_relapse_max_days": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "opt_pira_allow_relapses_between_event_and_confirmation": False,
        },
    ), "Test 4 failed!"
    # Test case 5 - (2, 10, True, 30, "last", True, "Last only, relapses in confirmation allowed")
    test_case_5_targets = {
        "days_since_previous_relapse": [(50, 5), (60, 15), (70, 25), (80, 35)],
        "days_to_next_relapse": [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        "is_post_event_rebaseline": [(30, True), (50, True)],
        "is_general_rebaseline": [(30, True), (50, True), (60, True)],
        "is_raw_pira_rebaseline": [(30, True), (50, True), (60, True)],
        "edss_score_used_as_new_general_reference": [(30, 2.0), (50, 3.0), (60, 2.5)],
        "edss_score_used_as_new_raw_pira_reference": [(30, 2.0), (50, 3.0), (60, 2.5)],
        "is_progression": [(30, True), (50, True)],
        "progression_type": [(30, LABEL_PIRA), (50, LABEL_RAW)],
        "progression_score": [(30, 2.0), (50, 3.0)],
        "progression_reference_score": [(30, 1.0), (50, 2.0)],
        "progression_event_id": [(30, 1), (50, 2)],
        "is_post_relapse_rebaseline": [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_5_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 2,
            "opt_raw_after_relapse_max_days": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "last",
            "opt_pira_allow_relapses_between_event_and_confirmation": True,
        },
    ), "Test 5 failed!"
    # Test case 6 - (2, 10, True, 30, "last", False, "Last only, relapses in confirmation interval not allowed")
    test_case_6_targets = {
        "days_since_previous_relapse": [(50, 5), (60, 15), (70, 25), (80, 35)],
        "days_to_next_relapse": [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        "is_post_event_rebaseline": [(30, True), (50, True)],
        "is_general_rebaseline": [(30, True), (50, True), (60, True)],
        "is_raw_pira_rebaseline": [(30, True), (50, True), (60, True)],
        "edss_score_used_as_new_general_reference": [(30, 2.0), (50, 3.0), (60, 2.5)],
        "edss_score_used_as_new_raw_pira_reference": [(30, 2.0), (50, 3.0), (60, 2.5)],
        "is_progression": [(30, True), (50, True)],
        "progression_type": [(30, LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW), (50, LABEL_RAW)],
        "progression_score": [(30, 2.0), (50, 3.0)],
        "progression_reference_score": [(30, 1.0), (50, 2.0)],
        "progression_event_id": [(30, 1), (50, 2)],
        "is_post_relapse_rebaseline": [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_6_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 2,
            "opt_raw_after_relapse_max_days": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "last",
            "opt_pira_allow_relapses_between_event_and_confirmation": False,
        },
    ), "Test 6 failed!"
    # Test case 7 - undefined candidate lower than RAW/PIRA must not be event
    """
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=False,
        relapse_timestamps=[25, 39],
        follow_up_dataframe=pd.DataFrame(
            {
                "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 70],
                "edss_score": [6.0, 6.5, 7.0, 7.0, 6.0, 6.5, 7.0, 7.0],
            }
        ),
        targets_dict={
            "days_to_next_relapse": [(0, 25), (10, 15), (20, 5), (30, 9)],
            "days_since_previous_relapse": [
                (30, 5),
                (40, 1),
                (50, 11),
                (60, 21),
                (70, 31),
            ],
            "is_post_event_rebaseline": [(60, True)],
            "is_general_rebaseline": [(60, True)],
            "edss_score_used_as_new_general_reference": [(60, 7)],
            "is_raw_pira_rebaseline": [(30, True), (40, True), (60, True)],
            "edss_score_used_as_new_raw_pira_reference": [(30, 7), (40, 6.5), (60, 7)],
            "is_post_relapse_rebaseline": [(30, True)],
            "is_progression": [(60, True)],
            "progression_type": [(60, LABEL_PIRA)],
            "progression_score": [(60, 7)],
            "progression_reference_score": [(60, 6.5)],
            "progression_event_id": [(60, 1)],
        },
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_raw_after_relapse_max_days": 3,
            "opt_raw_before_relapse_max_days": 3,
        },
    ), "Test 7 failed"
    """


def test_raw_pira_confirmed_mueller_2025():
    # Test 1 - confirm as PIRA despite relapse if later scores are present
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=False,
        relapse_timestamps=[45],
        follow_up_dataframe=pd.DataFrame(
            {
                "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 70, 80],
                "edss_score": [1, 1, 1.5, 2.0, 2.0, 3.0, 2.5, 2.5, 3.0],
            }
        ),
        targets_dict={
            "days_to_next_relapse": [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
            "days_since_previous_relapse": [
                (50, 5),
                (60, 15),
                (70, 25),
                (80, 35)
            ],
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2)],
            "is_raw_pira_rebaseline": [(30, True), (70, True)],
            "edss_score_used_as_new_raw_pira_reference": [(30, 2), (70, 2.5)],
            "is_post_relapse_rebaseline": [(70, True)],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 2)],
            "progression_reference_score": [(30, 1)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "opt_raw_before_relapse_max_days": 2,
            "opt_raw_after_relapse_max_days": 20,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse": True,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse_before_relapse_max_days": 2,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse_after_relapse_max_days": 20,
        },
    ), "Test 1 failed"
    # Test 2 - ignore post-relapse recovery
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=False,
        relapse_timestamps=[45],
        follow_up_dataframe=pd.DataFrame(
            {
                "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 70, 80],
                "edss_score": [1, 1, 1.5, 2.0, 2.0, 3.0, 1.5, 2.5, 3.0],
            }
        ),
        targets_dict={
            "days_to_next_relapse": [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
            "days_since_previous_relapse": [
                (50, 5),
                (60, 15),
                (70, 25),
                (80, 35)
            ],
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2)],
            "is_raw_pira_rebaseline": [(30, True), (70, True)],
            "edss_score_used_as_new_raw_pira_reference": [(30, 2), (70, 2.5)],
            "is_post_relapse_rebaseline": [(70, True)],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 2)],
            "progression_reference_score": [(30, 1)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "opt_raw_before_relapse_max_days": 2,
            "opt_raw_after_relapse_max_days": 20,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse": True,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse_before_relapse_max_days": 2,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse_after_relapse_max_days": 20,
        },
    ), "Test 2 failed"
    # Test 3 - don't confirm because of later post-relapse recovery
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=False,
        relapse_timestamps=[45],
        follow_up_dataframe=pd.DataFrame(
            {
                "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                "edss_score": [1, 1, 1.5, 2.0, 2.0, 3.0, 2.0, 1.5, 2.5, 3.0],
            }
        ),
        targets_dict={
            "days_to_next_relapse": [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
            "days_since_previous_relapse": [
                (50, 5),
                (60, 15),
                (70, 25),
                (80, 35),
                (90, 45),
            ],
            "is_raw_pira_rebaseline": [(70, True)],
            "edss_score_used_as_new_raw_pira_reference": [(70, 1.5)],
            "is_post_relapse_rebaseline": [(70, True)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "opt_raw_before_relapse_max_days": 2,
            "opt_raw_after_relapse_max_days": 20,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse": True,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse_before_relapse_max_days": 2,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse_after_relapse_max_days": 20,
        },
    ), "Test 3 failed"
    # Test 4 - leave events unconfirmed if no confirmation scores outside relapse window are present
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=False,
        relapse_timestamps=[45],
        follow_up_dataframe=pd.DataFrame(
            {
                "days_after_baseline": [0, 10, 20, 30, 40, 50, 60,],
                "edss_score": [1, 1, 1.5, 2.0, 2.0, 3.0, 2.5],
            }
        ),
        targets_dict={
            "days_to_next_relapse": [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
            "days_since_previous_relapse": [
                (50, 5),
                (60, 15),
            ],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "opt_raw_before_relapse_max_days": 2,
            "opt_raw_after_relapse_max_days": 20,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse": True,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse_before_relapse_max_days": 2,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse_after_relapse_max_days": 20,
        },
    ), "Test 4 failed"
    # Test 5 - different window sizes
    assert raw_pira_progression_result_is_equal_to_target(
        ignore_relapses=False,
        relapse_timestamps=[45],
        follow_up_dataframe=pd.DataFrame(
            {
                "days_after_baseline": [0, 10, 20, 30, 40, 50, 60],
                "edss_score": [1, 1, 1.5, 2.0, 2.0, 3.0, 2.5],
            }
        ),
        targets_dict={
            "days_to_next_relapse": [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
            "days_since_previous_relapse": [
                (50, 5),
                (60, 15),
            ],
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2)],
            "is_raw_pira_rebaseline": [(30, True)],
            "edss_score_used_as_new_raw_pira_reference": [(30, 2)],
            "is_progression": [(30, True)],
            "progression_type": [(30, LABEL_PIRA)],
            "progression_score": [(30, 2)],
            "progression_reference_score": [(30, 1)],
            "progression_event_id": [(30, 1)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "opt_raw_before_relapse_max_days": 2,
            "opt_raw_after_relapse_max_days": 20,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse": True,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse_before_relapse_max_days": 2,
            "opt_confirmation_pira_ignore_scores_in_proximity_to_relapse_after_relapse_max_days": 10,
        },
    ), "Test 5 failed"




def test_post_relapse_rebaselining_higher_equal_lower():
    test_relapse_timestamps = [15]
    common_targets = {
        "days_since_previous_relapse": [(20, 5), (30, 15), (40, 25), (50, 35)],
        "days_to_next_relapse": [(0, 15), (10, 5)],
        "is_general_rebaseline": [],
        "edss_score_used_as_new_general_reference": [],
        "is_progression": [],
        "progression_type": [],
        "progression_score": [],
        "progression_reference_score": [],
    }

    # Test case 1 - stable baseline
    test_case_1_dataframe = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50],
            "edss_score": [2.5, 3, 3, 2.5, 2.5, 2.5],
        }
    )
    test_case_1_targets = {
        "is_post_event_rebaseline": [],
        "is_raw_pira_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [],
        "is_post_relapse_rebaseline": [],
    }
    test_case_1_targets = {**test_case_1_targets, **common_targets}
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_case_1_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_1_targets,
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 1 failed!"

    # Test case 2 - residual disability
    test_case_2_dataframe = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50],
            "edss_score": [2.5, 3, 3, 3, 3, 3],
        }
    )
    test_case_2_targets = {
        "is_post_event_rebaseline": [],
        "is_raw_pira_rebaseline": [(30, True)],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3)],
        "is_post_relapse_rebaseline": [(30, True)],
    }
    test_case_2_targets = {**test_case_2_targets, **common_targets}
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_case_2_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_2_targets,
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 2 failed!"

    # Test case 3 - lower score after relapse
    test_case_3_dataframe = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50],
            "edss_score": [2.5, 3, 3, 2, 2, 2],
        }
    )
    test_case_3_targets = {
        "is_post_event_rebaseline": [],
        "is_raw_pira_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [],
        "is_post_relapse_rebaseline": [],
    }
    test_case_3_targets = {**test_case_3_targets, **common_targets}
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_case_3_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_3_targets,
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 3 failed!"

    # Test case 4 - event
    test_case_4_dataframe = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50],
            "edss_score": [2.5, 3.5, 3.5, 3, 3, 3],
        }
    )
    test_case_4_targets = {
        "days_since_previous_relapse": [(20, 5), (30, 15), (40, 25), (50, 35)],
        "days_to_next_relapse": [(0, 15), (10, 5)],
        "is_post_event_rebaseline": [(10, True)],
        "is_general_rebaseline": [(10, True)],
        "is_raw_pira_rebaseline": [(10, True)],
        "edss_score_used_as_new_general_reference": [(10, 3.5)],
        "edss_score_used_as_new_raw_pira_reference": [(10, 3.5)],
        "is_post_relapse_rebaseline": [],
        "is_progression": [(10, True)],
        "progression_type": [(10, LABEL_RAW)],
        "progression_score": [(10, 3.5)],
        "progression_reference_score": [(10, 2.5)],
        "progression_event_id": [(10, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_case_4_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_4_targets,
        args_dict={
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 4 failed!"


def test_post_relapse_rebaselining_backtracking():
    test_relapse_timestamps = [25]
    test_dataframe = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60],
            "edss_score": [3.5, 3, 2.5, 3, 4, 4, 4],
        }
    )

    # Test case 1 - with backtracking, stopped by post-relapse re-baselining
    test_case_1_targets = {
        "is_post_event_rebaseline": [(60, True)],
        "is_raw_pira_rebaseline": [(10, True), (20, True), (30, True), (60, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (10, 3),
            (20, 2.5),
            (30, 3),
            (60, 4),
        ],
        "is_post_relapse_rebaseline": [(30, True)],
        "days_since_previous_relapse": [(30, 5), (40, 15), (50, 25), (60, 35)],
        "days_to_next_relapse": [(0, 25), (10, 15), (20, 5)],
        "is_general_rebaseline": [(10, True), (20, True), (60, True)],
        "edss_score_used_as_new_general_reference": [(10, 3), (20, 2.5), (60, 4)],
        "is_progression": [(60, True)],
        "progression_type": [(60, LABEL_PIRA)],
        "progression_score": [(60, 4)],
        "progression_reference_score": [(60, 3.0)],
        "progression_event_id": [(60, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_1_targets,
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_raw_before_relapse_max_days": 3,
            "opt_raw_after_relapse_max_days": 3,
            "opt_require_confirmation": False,
            "opt_minimal_distance_time": 25,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 1 failed!"

    # Test case 2 - no residual disability after relapse
    test_case_2_targets = {
        "is_post_event_rebaseline": [(40, True)],
        "is_raw_pira_rebaseline": [(10, True), (20, True), (40, True)],
        "edss_score_used_as_new_raw_pira_reference": [(10, 3), (20, 2.5), (40, 4)],
        "is_post_relapse_rebaseline": [],
        "days_since_previous_relapse": [(30, 5), (40, 15), (50, 25), (60, 35)],
        "days_to_next_relapse": [(0, 25), (10, 15), (20, 5)],
        "is_general_rebaseline": [(10, True), (20, True), (40, True)],
        "edss_score_used_as_new_general_reference": [(10, 3), (20, 2.5), (40, 4)],
        "is_progression": [(40, True)],
        "progression_type": [(40, LABEL_PIRA)],
        "progression_score": [(40, 4)],
        "progression_reference_score": [(40, 3)],
        "progression_event_id": [(40, 1)],
    }
    test_dataframe.at[3, "edss_score"] = 2.5
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_2_targets,
        args_dict={
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_raw_before_relapse_max_days": 3,
            "opt_raw_after_relapse_max_days": 3,
            "opt_require_confirmation": False,
            "opt_minimal_distance_time": 25,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 2 failed!"


def test_post_relapse_rebaselining_with_missing_and_overlapping():
    test_relapses_cases_1_2_3_4 = [6, 30]
    test_dataframe_cases_1_2_3_4 = pd.DataFrame(
        {"days_after_baseline": [0, 8, 36, 48, 60], "edss_score": [1, 1.5, 2.5, 3, 3.5]}
    )
    test_common_targets_cases_1_2_3_4 = {
        "days_since_previous_relapse": [(8, 2), (36, 6), (48, 18), (60, 30)],
        "days_to_next_relapse": [(0, 6), (8, 22)],
    }

    # Test case 1 - assessments well-separated
    test_case_1_targets = {
        "is_post_event_rebaseline": [(36, True), (60, True)],
        "is_raw_pira_rebaseline": [(8, True), (36, True), (60, True)],
        "edss_score_used_as_new_raw_pira_reference": [(8, 1.5), (36, 2.5), (60, 3.5)],
        "is_post_relapse_rebaseline": [(8, True), (36, True)],
        "is_general_rebaseline": [(36, True), (60, True)],
        "edss_score_used_as_new_general_reference": [(36, 2.5), (60, 3.5)],
        "is_progression": [(36, True), (60, True)],
        "progression_type": [(36, LABEL_UNDEFINED_PROGRESSION), (60, LABEL_PIRA)],
        "progression_score": [(36, 2.5), (60, 3.5)],
        "progression_reference_score": [(36, 1.0), (60, 2.5)],
        "progression_event_id": [(36, 1), (60, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_1_2_3_4,
        relapse_timestamps=test_relapses_cases_1_2_3_4,
        targets_dict={**test_case_1_targets, **test_common_targets_cases_1_2_3_4},
        args_dict={
            "opt_raw_before_relapse_max_days": 2,
            "opt_raw_after_relapse_max_days": 1,
        },
    ), "Test 1 failed!"

    # Test case 2 - rebaseline of first relapse after second
    # relapse, within buffer
    test_case_2_targets = {
        "is_post_event_rebaseline": [(36, True)],
        "is_raw_pira_rebaseline": [(36, True), (48, True)],
        "edss_score_used_as_new_raw_pira_reference": [(36, 2.5), (48, 3)],
        "is_post_relapse_rebaseline": [(36, True), (48, True)],
        "is_general_rebaseline": [(36, True)],
        "edss_score_used_as_new_general_reference": [(36, 2.5)],
        "is_progression": [(36, True)],
        "progression_type": [(36, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(36, 2.5)],
        "progression_reference_score": [(36, 1.0)],
        "progression_event_id": [(36, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_1_2_3_4,
        relapse_timestamps=test_relapses_cases_1_2_3_4,
        targets_dict={**test_case_2_targets, **test_common_targets_cases_1_2_3_4},
        args_dict={
            "opt_raw_before_relapse_max_days": 4,
            "opt_raw_after_relapse_max_days": 12,
        },
    ), "Test 2 failed!"

    # Test case 3 - rebaseline of first relapse after second relapse,
    # after second buffer (same for both)
    test_case_3_targets = {
        "is_post_event_rebaseline": [(36, True), (60, True)],
        "is_raw_pira_rebaseline": [(36, True), (60, True)],
        "edss_score_used_as_new_raw_pira_reference": [(36, 2.5), (60, 3.5)],
        "is_post_relapse_rebaseline": [(36, True)],
        "is_general_rebaseline": [(36, True), (60, True)],
        "edss_score_used_as_new_general_reference": [(36, 2.5), (60, 3.5)],
        "is_progression": [(36, True), (60, True)],
        "progression_type": [(36, LABEL_UNDEFINED_PROGRESSION), (60, LABEL_PIRA)],
        "progression_score": [(36, 2.5), (60, 3.5)],
        "progression_reference_score": [(36, 1.0), (60, 2.5)],
        "progression_event_id": [(36, 1), (60, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_1_2_3_4,
        relapse_timestamps=test_relapses_cases_1_2_3_4,
        targets_dict={**test_case_3_targets, **test_common_targets_cases_1_2_3_4},
        args_dict={
            "opt_raw_before_relapse_max_days": 4,
            "opt_raw_after_relapse_max_days": 4,
        },
    ), "Test 3 failed!"

    # Test case 4 - overlapping RAW windows
    test_case_4_targets = {
        "is_post_event_rebaseline": [(36, True), (60, True)],
        "is_raw_pira_rebaseline": [(36, True), (60, True)],
        "edss_score_used_as_new_raw_pira_reference": [(36, 2.5), (60, 3.5)],
        "is_post_relapse_rebaseline": [(60, True)],
        "is_general_rebaseline": [(36, True), (60, True)],
        "edss_score_used_as_new_general_reference": [(36, 2.5), (60, 3.5)],
        "is_progression": [(36, True), (60, True)],
        "progression_type": [(36, LABEL_RAW), (60, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(36, 2.5), (60, 3.5)],
        "progression_reference_score": [(36, 1.0), (60, 2.5)],
        "progression_event_id": [(36, 1), (60, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_1_2_3_4,
        relapse_timestamps=test_relapses_cases_1_2_3_4,
        targets_dict={**test_case_4_targets, **test_common_targets_cases_1_2_3_4},
        args_dict={
            "opt_raw_before_relapse_max_days": 4,
            "opt_raw_after_relapse_max_days": 20,
        },
    ), "Test 4 failed!"

    # Test case 5 - rebaseline of first relapse before second, but within RAW window
    test_relapses_cases_5 = [6, 30]
    test_dataframe_cases_5 = pd.DataFrame(
        {"days_after_baseline": [0, 8, 28, 48, 60], "edss_score": [1, 1.5, 2.5, 3, 3.5]}
    )
    test_case_5_targets = {
        "days_since_previous_relapse": [(8, 2), (28, 22), (48, 18), (60, 30)],
        "days_to_next_relapse": [(0, 6), (8, 22), (28, 2)],
        "is_post_event_rebaseline": [(28, True)],
        "is_raw_pira_rebaseline": [(28, True), (48, True)],
        "edss_score_used_as_new_raw_pira_reference": [(28, 2.5), (48, 3)],
        "is_post_relapse_rebaseline": [(28, True), (48, True)],
        "is_general_rebaseline": [(28, True)],
        "edss_score_used_as_new_general_reference": [(28, 2.5)],
        "is_progression": [(28, True)],
        "progression_type": [(28, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(28, 2.5)],
        "progression_reference_score": [(28, 1.0)],
        "progression_event_id": [(28, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_5,
        relapse_timestamps=test_relapses_cases_5,
        targets_dict=test_case_5_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 4,
            "opt_raw_after_relapse_max_days": 4,
        },
    ), "Test 5 failed!"

    # Test case 6 - multiple non-overlapping and overlapping relapses
    test_relapses_cases_6 = [15, 25, 40, 48]
    test_dataframe_cases_6 = pd.DataFrame(
        {"days_after_baseline": [0, 10, 50, 60, 70], "edss_score": [1, 1, 2, 2, 2.5]}
    )
    test_case_6_targets = {
        "days_since_previous_relapse": [(50, 2), (60, 12), (70, 22)],
        "days_to_next_relapse": [(0, 15), (10, 5)],
        "is_post_event_rebaseline": [(50, True)],
        "is_raw_pira_rebaseline": [(50, True)],
        "edss_score_used_as_new_raw_pira_reference": [(50, 2)],
        "is_post_relapse_rebaseline": [(50, True)],
        "is_general_rebaseline": [(50, True)],
        "edss_score_used_as_new_general_reference": [(50, 2)],
        "is_progression": [(50, True)],
        "progression_type": [(50, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(50, 2)],
        "progression_reference_score": [(50, 1.0)],
        "progression_event_id": [(50, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_6,
        relapse_timestamps=test_relapses_cases_6,
        targets_dict=test_case_6_targets,
        args_dict={
            "opt_raw_before_relapse_max_days": 2,
            "opt_raw_after_relapse_max_days": 7,
        },
    ), "Test 6 failed!"


def test_post_relapse_rebaselining_with_event():
    test_dataframe = pd.DataFrame(
        {"days_after_baseline": [0, 20, 40, 60], "edss_score": [1, 1.5, 2.5, 2]}
    )
    test_relapses = [15, 25]

    test_common_targets = {
        "days_to_next_relapse": [(0, 15), (20, 5)],
        "days_since_previous_relapse": [(20, 5), (40, 15), (60, 35)],
        "is_post_event_rebaseline": [(40, True)],
        "is_raw_pira_rebaseline": [(20, True), (40, True)],
        "is_general_rebaseline": [(40, True)],
        "is_post_relapse_rebaseline": [(20, True), (40, True)],
        "is_progression": [(40, True)],
        "progression_type": [(40, LABEL_UNDEFINED_PROGRESSION)],
        "progression_event_id": [(40, 1)],
    }
    # Test case 1 - event unconfirmed
    test_case_1_targets = {
        "edss_score_used_as_new_raw_pira_reference": [(20, 1.5), (40, 2.5)],
        "edss_score_used_as_new_general_reference": [(40, 2.5)],
        "progression_score": [(40, 2.5)],
        "progression_reference_score": [(40, 1.0)],
        "progression_event_id": [(40, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapses,
        targets_dict={**test_case_1_targets, **test_common_targets},
        args_dict={
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0.5,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 4,
        },
    ), "Test 1 failed!"
    # Test case 2 - event confirmed
    test_case_2_targets = {
        "edss_score_used_as_new_raw_pira_reference": [(20, 1.5), (40, 2.5)],
        "edss_score_used_as_new_general_reference": [(40, 2)],
        "progression_score": [(40, 2)],
        "progression_reference_score": [(40, 1.0)],
        "progression_event_id": [(40, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapses,
        targets_dict={**test_case_2_targets, **test_common_targets},
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 4,
        },
    ), "Test 2 failed!"


def test_undefined_progression_never_option():
    test_dataframe = pd.DataFrame(
        {
            "days_after_baseline": [0, 20, 40, 60, 80],
            "edss_score": [1, 1.5, 2.5, 3.5, 4.5],
        }
    )
    test_common_targets = {
        "days_to_next_relapse": [(0, 15)],
        "days_since_previous_relapse": [(20, 5), (40, 25), (60, 45), (80, 65)],
    }

    # Test case 1 - allow undefined at re-baselining, all events
    test_case_1_targets = {
        "is_post_event_rebaseline": [(40, True), (60, True), (80, True)],
        "is_raw_pira_rebaseline": [(40, True), (60, True), (80, True)],
        "is_general_rebaseline": [(40, True), (60, True), (80, True)],
        "is_post_relapse_rebaseline": [(40, True)],
        "edss_score_used_as_new_raw_pira_reference": [(40, 2.5), (60, 3.5), (80, 4.5)],
        "edss_score_used_as_new_general_reference": [(40, 2.5), (60, 3.5), (80, 4.5)],
        "is_progression": [(40, True), (60, True), (80, True)],
        "progression_type": [
            (40, LABEL_UNDEFINED_PROGRESSION),
            (60, LABEL_PIRA),
            (80, LABEL_PIRA),
        ],
        "progression_score": [(40, 2.5), (60, 3.5), (80, 4.5)],
        "progression_reference_score": [(40, 1.0), (60, 2.5), (80, 3.5)],
        "progression_event_id": [(40, 1), (60, 2), (80, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict={**test_common_targets, **test_case_1_targets},
        args_dict={
            "undefined_progression": "re-baselining only",
            "return_first_event_only": False,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
        },
    ), "Test 1 failed!"

    # Test case 2 - allow undefined at re-baselining, first event only
    test_case_2_targets = {
        "is_post_relapse_rebaseline": [(40, True)],
        "is_progression": [(40, True)],
        "progression_type": [(40, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(40, 2.5)],
        "progression_reference_score": [(40, 1.0)],
        "progression_event_id": [(40, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict={**test_common_targets, **test_case_2_targets},
        args_dict={
            "undefined_progression": "re-baselining only",
            "return_first_event_only": True,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
        },
    ), "Test 2 failed!"

    # Test case 3 - don't allow undefined at re-baselining, all events
    test_case_3_targets = {
        "is_post_event_rebaseline": [(60, True), (80, True)],
        "is_raw_pira_rebaseline": [(40, True), (60, True), (80, True)],
        "is_general_rebaseline": [(60, True), (80, True)],
        "is_post_relapse_rebaseline": [(40, True)],
        "edss_score_used_as_new_raw_pira_reference": [(40, 2.5), (60, 3.5), (80, 4.5)],
        "edss_score_used_as_new_general_reference": [(60, 3.5), (80, 4.5)],
        "is_progression": [(60, True), (80, True)],
        "progression_type": [(60, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(60, 3.5), (80, 4.5)],
        "progression_reference_score": [(60, 2.5), (80, 3.5)],
        "progression_event_id": [(60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict={**test_common_targets, **test_case_3_targets},
        args_dict={
            "undefined_progression": "never",
            "return_first_event_only": False,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
        },
    ), "Test 3 failed!"

    # Test case 4 - don't allow undefined at re-baselining, last event only
    test_case_4_targets = {
        "is_raw_pira_rebaseline": [(40, True)],
        "is_post_relapse_rebaseline": [(40, True)],
        "edss_score_used_as_new_raw_pira_reference": [(40, 2.5)],
        "is_progression": [(60, True)],
        "progression_type": [(60, LABEL_PIRA)],
        "progression_score": [(60, 3.5)],
        "progression_reference_score": [(60, 2.5)],
        "progression_event_id": [(60, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict={**test_common_targets, **test_case_4_targets},
        args_dict={
            "undefined_progression": "never",
            "return_first_event_only": True,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
        },
    ), "Test 4 failed!"

    with_confirmation_test_dataframe = pd.DataFrame(
        {"days_after_baseline": [0, 20, 40, 60], "edss_score": [1, 1.5, 2.5, 2]}
    )
    with_confirmation_test_common_targets = {
        "days_to_next_relapse": [(0, 15), (20, 5)],
        "days_since_previous_relapse": [(20, 5), (40, 15), (60, 35)],
    }
    # Test case 5 - allow undefined at re-baselining, all events
    test_case_5_targets = {
        "is_post_event_rebaseline": [(40, True)],
        "is_raw_pira_rebaseline": [(20, True), (40, True)],
        "is_general_rebaseline": [(40, True)],
        "is_post_relapse_rebaseline": [(20, True), (40, True)],
        "edss_score_used_as_new_raw_pira_reference": [(20, 1.5), (40, 2.5)],
        "edss_score_used_as_new_general_reference": [(40, 2.0)],
        "is_progression": [(40, True)],
        "progression_type": [(40, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(40, 2.0)],
        "progression_reference_score": [(40, 1.0)],
        "progression_event_id": [(40, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=with_confirmation_test_dataframe,
        relapse_timestamps=[15, 25],
        targets_dict={**with_confirmation_test_common_targets, **test_case_5_targets},
        args_dict={
            "undefined_progression": "re-baselining only",
            "return_first_event_only": False,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 5 failed!"
    # Test case 6 - do not allow undefined at re-baselining, all events
    test_case_6_targets = {
        "is_raw_pira_rebaseline": [(20, True), (40, True)],
        "is_post_relapse_rebaseline": [(20, True), (40, True)],
        "edss_score_used_as_new_raw_pira_reference": [(20, 1.5), (40, 2.5)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=with_confirmation_test_dataframe,
        relapse_timestamps=[15, 25],
        targets_dict={**with_confirmation_test_common_targets, **test_case_6_targets},
        args_dict={
            "undefined_progression": "never",
            "return_first_event_only": False,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 6 failed!"
    # Test case 7 - 'never' option without relapses
    test_case_7_targets = {
        "is_post_event_rebaseline": [(40, True), (60, True), (80, True)],
        "is_raw_pira_rebaseline": [(40, True), (60, True), (80, True)],
        "is_general_rebaseline": [(40, True), (60, True), (80, True)],
        "edss_score_used_as_new_raw_pira_reference": [(40, 2.5), (60, 3.5), (80, 4.5)],
        "edss_score_used_as_new_general_reference": [(40, 2.5), (60, 3.5), (80, 4.5)],
        "is_progression": [(40, True), (60, True), (80, True)],
        "progression_type": [(40, LABEL_PIRA), (60, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(40, 2.5), (60, 3.5), (80, 4.5)],
        "progression_reference_score": [(40, 1.0), (60, 2.5), (80, 3.5)],
        "progression_event_id": [(40, 1), (60, 2), (80, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[],
        targets_dict=test_case_7_targets,
        args_dict={
            "undefined_progression": "never",
            "return_first_event_only": False,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
        },
    ), "Test 7 failed!"


def test_undefined_progression_all_option():
    # Series 1 - default, never, all
    test_dataframe_series_1 = pd.DataFrame(
        {
            "days_after_baseline": [i * 10 for i in range(17)],
            "edss_score": [
                1,
                2.5,
                3.0,
                1.5,
                3.5,
                2,
                2,
                2,
                1,
                1,
                4.5,
                2.5,
                2,
                3,
                3.5,
                4,
                4.5,
            ],
        }
    )
    test_relapse_timestamps_series_1 = [5, 35, 95]
    test_common_targets_series_1 = {
        "days_to_next_relapse": [
            (0, 5),
            (10, 25),
            (20, 15),
            (30, 5),
            (40, 55),
            (50, 45),
            (60, 35),
            (70, 25),
            (80, 15),
            (90, 5),
        ],
        "days_since_previous_relapse": [
            (10, 5),
            (20, 15),
            (30, 25),
            (40, 5),
            (50, 15),
            (60, 25),
            (70, 35),
            (80, 45),
            (90, 55),
            (100, 5),
            (110, 15),
            (120, 25),
            (130, 35),
            (140, 45),
            (150, 55),
            (160, 65),
        ],
    }
    # Test case 1 - allow undefined at re-baselining, all events
    test_case_1_targets = {
        "is_post_event_rebaseline": [(10, True), (100, True), (140, True), (160, True)],
        "is_raw_pira_rebaseline": [
            (10, True),
            (20, True),
            (50, True),
            (80, True),
            (100, True),
            (110, True),
            (140, True),
            (160, True),
        ],
        "is_general_rebaseline": [
            (10, True),
            (50, True),
            (80, True),
            (100, True),
            (110, True),
            (140, True),
            (160, True),
        ],
        "is_post_relapse_rebaseline": [(20, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (10, 2.5),
            (20, 3),
            (50, 2),
            (80, 1),
            (100, 4.5),
            (110, 2.5),
            (140, 3.5),
            (160, 4.5),
        ],
        "edss_score_used_as_new_general_reference": [
            (10, 2.5),
            (50, 2),
            (80, 1),
            (100, 4.5),
            (110, 2.5),
            (140, 3.5),
            (160, 4.5),
        ],
        "is_progression": [(10, True), (100, True), (140, True), (160, True)],
        "progression_type": [
            (10, LABEL_RAW),
            (100, LABEL_RAW),
            (140, LABEL_PIRA),
            (160, LABEL_PIRA),
        ],
        "progression_score": [(10, 2.5), (100, 4.5), (140, 3.5), (160, 4.5)],
        "progression_reference_score": [(10, 1.0), (100, 1.0), (140, 2.5), (160, 3.5)],
        "progression_event_id": [(10, 1), (100, 2), (140, 3), (160, 4)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_1,
        relapse_timestamps=test_relapse_timestamps_series_1,
        targets_dict={**test_common_targets_series_1, **test_case_1_targets},
        args_dict={
            "undefined_progression": "re-baselining only",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
        },
    ), "Test 1 failed!"
    # Test case 2 - never allow undefined, all events
    test_case_2_targets = {
        "is_post_event_rebaseline": [(10, True), (100, True), (140, True), (160, True)],
        "is_raw_pira_rebaseline": [
            (10, True),
            (20, True),
            (50, True),
            (80, True),
            (100, True),
            (110, True),
            (140, True),
            (160, True),
        ],
        "is_general_rebaseline": [
            (10, True),
            (50, True),
            (80, True),
            (100, True),
            (110, True),
            (140, True),
            (160, True),
        ],
        "is_post_relapse_rebaseline": [(20, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (10, 2.5),
            (20, 3),
            (50, 2),
            (80, 1),
            (100, 4.5),
            (110, 2.5),
            (140, 3.5),
            (160, 4.5),
        ],
        "edss_score_used_as_new_general_reference": [
            (10, 2.5),
            (50, 2),
            (80, 1),
            (100, 4.5),
            (110, 2.5),
            (140, 3.5),
            (160, 4.5),
        ],
        "is_progression": [(10, True), (100, True), (140, True), (160, True)],
        "progression_type": [
            (10, LABEL_RAW),
            (100, LABEL_RAW),
            (140, LABEL_PIRA),
            (160, LABEL_PIRA),
        ],
        "progression_score": [(10, 2.5), (100, 4.5), (140, 3.5), (160, 4.5)],
        "progression_reference_score": [(10, 1.0), (100, 1.0), (140, 2.5), (160, 3.5)],
        "progression_event_id": [(10, 1), (100, 2), (140, 3), (160, 4)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_1,
        relapse_timestamps=test_relapse_timestamps_series_1,
        targets_dict={**test_common_targets_series_1, **test_case_2_targets},
        args_dict={
            "undefined_progression": "never",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
        },
    ), "Test 2 failed!"
    # Test case 3 - allow undefined at all assessments
    test_case_3_targets = {
        "is_post_event_rebaseline": [
            (10, True),
            (40, True),
            (100, True),
            (140, True),
            (160, True),
        ],
        "is_raw_pira_rebaseline": [
            (10, True),
            (20, True),
            (40, True),
            (50, True),
            (80, True),
            (100, True),
            (110, True),
            (140, True),
            (160, True),
        ],
        "is_general_rebaseline": [
            (10, True),
            (40, True),
            (50, True),
            (80, True),
            (100, True),
            (110, True),
            (140, True),
            (160, True),
        ],
        "is_post_relapse_rebaseline": [(20, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (10, 2.5),
            (20, 3),
            (40, 3.5),
            (50, 2),
            (80, 1),
            (100, 4.5),
            (110, 2.5),
            (140, 3.5),
            (160, 4.5),
        ],
        "edss_score_used_as_new_general_reference": [
            (10, 2.5),
            (40, 3.5),
            (50, 2),
            (80, 1),
            (100, 4.5),
            (110, 2.5),
            (140, 3.5),
            (160, 4.5),
        ],
        "is_progression": [
            (10, True),
            (40, True),
            (100, True),
            (140, True),
            (160, True),
        ],
        "progression_type": [
            (10, LABEL_RAW),
            (40, LABEL_UNDEFINED_PROGRESSION),
            (100, LABEL_RAW),
            (140, LABEL_PIRA),
            (160, LABEL_PIRA),
        ],
        "progression_score": [(10, 2.5), (40, 3.5), (100, 4.5), (140, 3.5), (160, 4.5)],
        "progression_reference_score": [
            (10, 1.0),
            (40, 2.5),
            (100, 1.0),
            (140, 2.5),
            (160, 3.5),
        ],
        "progression_event_id": [(10, 1), (40, 2), (100, 3), (140, 4), (160, 5)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_1,
        relapse_timestamps=test_relapse_timestamps_series_1,
        targets_dict={**test_common_targets_series_1, **test_case_3_targets},
        args_dict={
            "undefined_progression": "all",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
        },
    ), "Test 3 failed!"

    # Series 2 - effect on RAW window size
    test_dataframe_series_2 = pd.DataFrame(
        {
            "days_after_baseline": [0, 65, 85, 105, 140, 230],
            "edss_score": [1, 2, 1.5, 1.5, 2, 2],
        }
    )
    test_relapse_timestamps_series_2 = [40]
    test_common_targets_series_2 = {
        "days_to_next_relapse": [(0, 40)],
        "days_since_previous_relapse": [
            (65, 25),
            (85, 45),
            (105, 65),
            (140, 100),
            (230, 190),
        ],
    }
    # Test case 4 - default, confirmed, 30 days post-relapse
    test_case_4_targets = {
        "is_post_event_rebaseline": [],
        "is_raw_pira_rebaseline": [(85, True)],
        "is_general_rebaseline": [],
        "is_post_relapse_rebaseline": [(85, True)],
        "edss_score_used_as_new_raw_pira_reference": [(85, 1.5)],
        "edss_score_used_as_new_general_reference": [],
        "is_progression": [],
        "progression_type": [],
        "progression_score": [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_2,
        relapse_timestamps=test_relapse_timestamps_series_2,
        targets_dict={**test_common_targets_series_2, **test_case_4_targets},
        args_dict={
            "undefined_progression": "re-baselining only",
            "opt_raw_before_relapse_max_days": 30,
            "opt_raw_after_relapse_max_days": 30,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 4 failed!"
    # Test case 5 - default, confirmed, 90 days post-relapse
    test_case_5_targets = {
        "is_post_event_rebaseline": [(140, True)],
        "is_raw_pira_rebaseline": [(140, True)],
        "is_general_rebaseline": [(140, True)],
        "is_post_relapse_rebaseline": [(140, True)],
        "edss_score_used_as_new_raw_pira_reference": [(140, 2)],
        "edss_score_used_as_new_general_reference": [(140, 2)],
        "is_progression": [(140, True)],
        "progression_type": [(140, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(140, 2)],
        "progression_reference_score": [(140, 1.0)],
        "progression_event_id": [(140, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_2,
        relapse_timestamps=test_relapse_timestamps_series_2,
        targets_dict={**test_common_targets_series_2, **test_case_5_targets},
        args_dict={
            "undefined_progression": "re-baselining only",
            "opt_raw_before_relapse_max_days": 30,
            "opt_raw_after_relapse_max_days": 90,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 5 failed!"
    # Test case 6 - all, confirmed, 30 days post-relapse
    test_case_6_targets = {
        "is_post_event_rebaseline": [(140, True)],
        "is_raw_pira_rebaseline": [(85, True), (140, True)],
        "is_general_rebaseline": [(140, True)],
        "is_post_relapse_rebaseline": [(85, True)],
        "edss_score_used_as_new_raw_pira_reference": [(85, 1.5), (140, 2)],
        "edss_score_used_as_new_general_reference": [(140, 2)],
        "is_progression": [(140, True)],
        "progression_type": [(140, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(140, 2)],
        "progression_reference_score": [(140, 1.0)],
        "progression_event_id": [(140, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_2,
        relapse_timestamps=test_relapse_timestamps_series_2,
        targets_dict={**test_common_targets_series_2, **test_case_6_targets},
        args_dict={
            "undefined_progression": "all",
            "opt_raw_before_relapse_max_days": 30,
            "opt_raw_after_relapse_max_days": 30,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 6 failed!"
    # Test case 7 - all, confirmed, 90 days post-relapse
    test_case_7_targets = {
        "is_post_event_rebaseline": [(140, True)],
        "is_raw_pira_rebaseline": [(140, True)],
        "is_general_rebaseline": [(140, True)],
        "is_post_relapse_rebaseline": [(140, True)],
        "edss_score_used_as_new_raw_pira_reference": [(140, 2)],
        "edss_score_used_as_new_general_reference": [(140, 2)],
        "is_progression": [(140, True)],
        "progression_type": [(140, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(140, 2)],
        "progression_reference_score": [(140, 1.0)],
        "progression_event_id": [(140, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_2,
        relapse_timestamps=test_relapse_timestamps_series_2,
        targets_dict={**test_common_targets_series_2, **test_case_7_targets},
        args_dict={
            "undefined_progression": "all",
            "opt_raw_before_relapse_max_days": 30,
            "opt_raw_after_relapse_max_days": 90,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 7 failed!"

    # Series 3 - masking of PIRA
    test_dataframe_series_3 = pd.DataFrame(
        {
            "days_after_baseline": [0, 65, 85, 105, 140, 200, 260],
            "edss_score": [1, 2, 1.5, 1.5, 2, 2.5, 2.5],
        }
    )
    test_relapse_timestamps_series_3 = [40]
    test_common_targets_series_3 = {
        "days_to_next_relapse": [(0, 40)],
        "days_since_previous_relapse": [
            (65, 25),
            (85, 45),
            (105, 65),
            (140, 100),
            (200, 160),
            (260, 220),
        ],
    }
    # Test case 8 - default, confirmed
    test_case_8_targets = {
        "is_post_event_rebaseline": [(200, True)],
        "is_raw_pira_rebaseline": [(85, True), (200, True)],
        "is_general_rebaseline": [(200, True)],
        "is_post_relapse_rebaseline": [(85, True)],
        "edss_score_used_as_new_raw_pira_reference": [(85, 1.5), (200, 2.5)],
        "edss_score_used_as_new_general_reference": [(200, 2.5)],
        "is_progression": [(200, True)],
        "progression_type": [(200, LABEL_PIRA)],
        "progression_score": [(200, 2.5)],
        "progression_reference_score": [(200, 1.5)],
        "progression_event_id": [(200, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_3,
        relapse_timestamps=test_relapse_timestamps_series_3,
        targets_dict={**test_common_targets_series_3, **test_case_8_targets},
        args_dict={
            "undefined_progression": "re-baselining only",
            "opt_raw_before_relapse_max_days": 30,
            "opt_raw_after_relapse_max_days": 30,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 8 failed!"
    # Test case 9 - all, confirmed
    test_case_9_targets = {
        "is_post_event_rebaseline": [(140, True)],
        "is_raw_pira_rebaseline": [(85, True), (140, True)],
        "is_general_rebaseline": [(140, True)],
        "is_post_relapse_rebaseline": [(85, True)],
        "edss_score_used_as_new_raw_pira_reference": [(85, 1.5), (140, 2)],
        "edss_score_used_as_new_general_reference": [(140, 2)],
        "is_progression": [(140, True)],
        "progression_type": [(140, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(140, 2)],
        "progression_reference_score": [(140, 1.0)],
        "progression_event_id": [(140, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_3,
        relapse_timestamps=test_relapse_timestamps_series_3,
        targets_dict={**test_common_targets_series_3, **test_case_9_targets},
        args_dict={
            "undefined_progression": "all",
            "opt_raw_before_relapse_max_days": 30,
            "opt_raw_after_relapse_max_days": 30,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 9 failed!"

    # Series 4 - masking of RAW
    test_dataframe_series_4 = pd.DataFrame(
        {
            "days_after_baseline": [0, 65, 85, 105, 140, 170, 200, 260],
            "edss_score": [1, 2, 1.5, 1.5, 2, 2.5, 2.5, 2.5],
        }
    )
    test_relapse_timestamps_series_4 = [40, 145]
    test_common_targets_series_4 = {
        "days_to_next_relapse": [(0, 40), (65, 80), (85, 60), (105, 40), (140, 5)],
        "days_since_previous_relapse": [
            (65, 25),
            (85, 45),
            (105, 65),
            (140, 100),
            (170, 25),
            (200, 55),
            (260, 115),
        ],
    }
    # Test case 10 - default, confirmed
    test_case_10_targets = {
        "is_post_event_rebaseline": [(170, True)],
        "is_raw_pira_rebaseline": [(85, True), (170, True)],
        "is_general_rebaseline": [(170, True)],
        "is_post_relapse_rebaseline": [(85, True)],
        "edss_score_used_as_new_raw_pira_reference": [(85, 1.5), (170, 2.5)],
        "edss_score_used_as_new_general_reference": [(170, 2.5)],
        "is_progression": [(170, True)],
        "progression_type": [(170, LABEL_RAW)],
        "progression_score": [(170, 2.5)],
        "progression_reference_score": [(170, 1.5)],
        "progression_event_id": [(170, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_4,
        relapse_timestamps=test_relapse_timestamps_series_4,
        targets_dict={**test_common_targets_series_4, **test_case_10_targets},
        args_dict={
            "undefined_progression": "re-baselining only",
            "opt_raw_before_relapse_max_days": 30,
            "opt_raw_after_relapse_max_days": 30,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 10 failed!"
    # Test case 11 - default, confirmed
    test_case_11_targets = {
        "is_post_event_rebaseline": [(140, True)],
        "is_raw_pira_rebaseline": [(85, True), (140, True), (200, True)],
        "is_general_rebaseline": [(140, True)],
        "is_post_relapse_rebaseline": [(85, True), (200, True)],
        "edss_score_used_as_new_raw_pira_reference": [(85, 1.5), (140, 2), (200, 2.5)],
        "edss_score_used_as_new_general_reference": [(140, 2)],
        "is_progression": [(140, True)],
        "progression_type": [(140, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(140, 2)],
        "progression_reference_score": [(140, 1.0)],
        "progression_event_id": [(140, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_4,
        relapse_timestamps=test_relapse_timestamps_series_4,
        targets_dict={**test_common_targets_series_4, **test_case_11_targets},
        args_dict={
            "undefined_progression": "all",
            "opt_raw_before_relapse_max_days": 30,
            "opt_raw_after_relapse_max_days": 30,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 11 failed!"

    # Test series 5 - lower confirmed score
    test_dataframe_series_5 = pd.DataFrame(
        {
            "days_after_baseline": [0, 20, 40, 60, 80, 100, 120],
            "edss_score": [1, 1.5, 2.5, 2, 3, 3.5, 3.5],
        }
    )
    test_relapse_timestamps_series_5 = [15, 25]
    test_common_targets_series_5 = {
        "days_to_next_relapse": [(0, 15), (20, 5)],
        "days_since_previous_relapse": [
            (20, 5),
            (40, 15),
            (60, 35),
            (80, 55),
            (100, 75),
            (120, 95),
        ],
    }
    # Test case 12 - default, confirmed
    test_case_12_targets = {
        "is_post_event_rebaseline": [(40, True), (100, True)],
        "is_raw_pira_rebaseline": [(20, True), (40, True), (100, True)],
        "is_general_rebaseline": [(40, True), (100, True)],
        "is_post_relapse_rebaseline": [(20, True), (40, True)],
        "edss_score_used_as_new_raw_pira_reference": [(20, 1.5), (40, 2.5), (100, 3.5)],
        "edss_score_used_as_new_general_reference": [(40, 2), (100, 3.5)],
        "is_progression": [(40, True), (100, True)],
        "progression_type": [(40, LABEL_UNDEFINED_PROGRESSION), (100, LABEL_PIRA)],
        "progression_score": [(40, 2), (100, 3.5)],
        "progression_reference_score": [(40, 1.0), (100, 2.5)],
        "progression_event_id": [(40, 1), (100, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_5,
        relapse_timestamps=test_relapse_timestamps_series_5,
        targets_dict={**test_common_targets_series_5, **test_case_12_targets},
        args_dict={
            "undefined_progression": "re-baselining only",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 12 failed!"
    # Test case 13 - all, confirmed
    test_case_13_targets = {
        "is_post_event_rebaseline": [(40, True), (80, True)],
        "is_raw_pira_rebaseline": [(20, True), (40, True), (80, True)],
        "is_general_rebaseline": [(40, True), (80, True)],
        "is_post_relapse_rebaseline": [(20, True), (40, True)],
        "edss_score_used_as_new_raw_pira_reference": [(20, 1.5), (40, 2.5), (80, 3)],
        "edss_score_used_as_new_general_reference": [(40, 2), (80, 3)],
        "is_progression": [(40, True), (80, True)],
        "progression_type": [
            (40, LABEL_UNDEFINED_PROGRESSION),
            (80, LABEL_UNDEFINED_PROGRESSION),
        ],
        "progression_score": [(40, 2), (80, 3)],
        "progression_reference_score": [(40, 1), (80, 2)],
        "progression_event_id": [(40, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_series_5,
        relapse_timestamps=test_relapse_timestamps_series_5,
        targets_dict={**test_common_targets_series_5, **test_case_13_targets},
        args_dict={
            "undefined_progression": "all",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 13 failed!"


def test_undefined_progression_end_option():
    # Test case 1 - previous RAW, earlier UP
    test_dataframe_case_1 = pd.DataFrame(
        {
            "days_after_baseline": [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                140,
                150,
                160,
                170,
            ],
            "edss_score": [0, 1, 2, 1.5, 1.5, 2, 2, 1.5, 1.5, 2.5, 2, 2, 2.5, 2.5],
        }
    )
    test_relapse_timestamps_case_1 = [15, 85]
    # Test case 1 - allow undefined at re-baselining, all events
    test_case_1_targets = {
        "days_to_next_relapse": [
            (0, 15),
            (10, 5),
            (20, 65),
            (30, 55),
            (40, 45),
            (50, 35),
            (60, 25),
            (70, 15),
            (80, 5),
        ],
        "days_since_previous_relapse": [
            (20, 5),
            (30, 15),
            (40, 25),
            (50, 35),
            (60, 45),
            (70, 55),
            (80, 65),
            (90, 5),
            (140, 55),
            (150, 65),
            (160, 75),
            (170, 85),
        ],
        "is_post_event_rebaseline": [(10, True), (50, True), (160, True)],
        "is_raw_pira_rebaseline": [
            (10, True),
            (30, True),
            (50, True),
            (70, True),
            (140, True),
            (160, True),
        ],
        "is_general_rebaseline": [(10, True), (50, True), (70, True), (160, True)],
        "is_post_relapse_rebaseline": [(30, True), (140, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (10, 1),
            (30, 1.5),
            (50, 2),
            (70, 1.5),
            (140, 2),
            (160, 2.5),
        ],
        "edss_score_used_as_new_general_reference": [
            (10, 1),
            (50, 2),
            (70, 1.5),
            (160, 2.5),
        ],
        "is_progression": [(10, True), (50, True), (160, True)],
        "progression_type": [
            (10, LABEL_RAW),
            (50, LABEL_UNDEFINED_PROGRESSION),
            (160, LABEL_UNDEFINED_PROGRESSION),
        ],
        "progression_score": [(10, 1), (50, 2), (160, 2.5)],
        "progression_reference_score": [(10, 0), (50, 1), (160, 1.5)],
        "progression_event_id": [(10, 1), (50, 2), (160, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_1,
        relapse_timestamps=test_relapse_timestamps_case_1,
        targets_dict=test_case_1_targets,
        args_dict={
            "undefined_progression": "end",
            "opt_larger_increment_from_0": False,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
        },
    ), "Test 1 failed!"
    # Test case 2 - without RAW at the beginning
    test_case_2_targets = {
        "days_to_next_relapse": [
            (10, 5),
            (20, 65),
            (30, 55),
            (40, 45),
            (50, 35),
            (60, 25),
            (70, 15),
            (80, 5),
        ],
        "days_since_previous_relapse": [
            (20, 5),
            (30, 15),
            (40, 25),
            (50, 35),
            (60, 45),
            (70, 55),
            (80, 65),
            (90, 5),
            (140, 55),
            (150, 65),
            (160, 75),
            (170, 85),
        ],
        "is_post_event_rebaseline": [(50, True), (160, True)],
        "is_raw_pira_rebaseline": [
            (30, True),
            (50, True),
            (70, True),
            (140, True),
            (160, True),
        ],
        "is_general_rebaseline": [(50, True), (70, True), (160, True)],
        "is_post_relapse_rebaseline": [(30, True), (140, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (30, 1.5),
            (50, 2),
            (70, 1.5),
            (140, 2),
            (160, 2.5),
        ],
        "edss_score_used_as_new_general_reference": [(50, 2), (70, 1.5), (160, 2.5)],
        "is_progression": [(50, True), (160, True)],
        "progression_type": [
            (50, LABEL_UNDEFINED_PROGRESSION),
            (160, LABEL_UNDEFINED_PROGRESSION),
        ],
        "progression_score": [(50, 2), (160, 2.5)],
        "progression_reference_score": [(50, 1), (160, 1.5)],
        "progression_event_id": [(50, 1), (160, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_1.iloc[1:].reset_index(drop=True),
        relapse_timestamps=test_relapse_timestamps_case_1,
        targets_dict=test_case_2_targets,
        args_dict={
            "undefined_progression": "end",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
        },
    ), "Test 2 failed!"
    # Test case 3 - masking of PIRA
    test_dataframe_case_3 = pd.DataFrame(
        {
            "days_after_baseline": [0, 65, 85, 105, 140, 200, 260],
            "edss_score": [1, 2, 1.5, 1.5, 2, 2.5, 2.5],
        }
    )
    test_relapse_timestamps_case_3 = [40]
    test_case_3_targets = {
        "days_to_next_relapse": [(0, 40)],
        "days_since_previous_relapse": [
            (65, 25),
            (85, 45),
            (105, 65),
            (140, 100),
            (200, 160),
            (260, 220),
        ],
        "is_post_event_rebaseline": [(200, True)],
        "is_raw_pira_rebaseline": [(85, True), (200, True)],
        "is_general_rebaseline": [(200, True)],
        "is_post_relapse_rebaseline": [(85, True)],
        "edss_score_used_as_new_raw_pira_reference": [(85, 1.5), (200, 2.5)],
        "edss_score_used_as_new_general_reference": [(200, 2.5)],
        "is_progression": [(200, True)],
        "progression_type": [(200, LABEL_PIRA)],
        "progression_score": [(200, 2.5)],
        "progression_reference_score": [(200, 1.5)],
        "progression_event_id": [(200, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_3,
        relapse_timestamps=test_relapse_timestamps_case_3,
        targets_dict=test_case_3_targets,
        args_dict={
            "undefined_progression": "end",
            "opt_raw_before_relapse_max_days": 30,
            "opt_raw_after_relapse_max_days": 30,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 3 failed!"
    # Test case 4 - masking of RAW
    test_dataframe_case_4 = pd.DataFrame(
        {
            "days_after_baseline": [0, 65, 85, 105, 140, 170, 200, 260],
            "edss_score": [1, 2, 1.5, 1.5, 2, 2.5, 2.5, 2.5],
        }
    )
    test_relapse_timestamps_case_4 = [40, 145]
    test_case_4_targets = {
        "days_to_next_relapse": [(0, 40), (65, 80), (85, 60), (105, 40), (140, 5)],
        "days_since_previous_relapse": [
            (65, 25),
            (85, 45),
            (105, 65),
            (140, 100),
            (170, 25),
            (200, 55),
            (260, 115),
        ],
        "is_post_event_rebaseline": [(170, True)],
        "is_raw_pira_rebaseline": [(85, True), (170, True)],
        "is_general_rebaseline": [(170, True)],
        "is_post_relapse_rebaseline": [(85, True)],
        "edss_score_used_as_new_raw_pira_reference": [(85, 1.5), (170, 2.5)],
        "edss_score_used_as_new_general_reference": [(170, 2.5)],
        "is_progression": [(170, True)],
        "progression_type": [(170, LABEL_RAW)],
        "progression_score": [(170, 2.5)],
        "progression_reference_score": [(170, 1.5)],
        "progression_event_id": [(170, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_4,
        relapse_timestamps=test_relapse_timestamps_case_4,
        targets_dict=test_case_4_targets,
        args_dict={
            "undefined_progression": "end",
            "opt_raw_before_relapse_max_days": 30,
            "opt_raw_after_relapse_max_days": 30,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 4 failed!"
    # Test case 5 - UP within RAW
    test_dataframe_case_5 = pd.DataFrame(
        {
            "days_after_baseline": [i * 10 for i in range(7)],
            "edss_score": [1, 2.5, 3.0, 1.5, 3.5, 2, 2],
        }
    )
    test_case_5_targets = {
        "days_to_next_relapse": [(0, 5), (10, 25), (20, 15), (30, 5)],
        "days_since_previous_relapse": [
            (10, 5),
            (20, 15),
            (30, 25),
            (40, 5),
            (50, 15),
            (60, 25),
        ],
        "is_post_event_rebaseline": [(10, True), (40, True)],
        "is_raw_pira_rebaseline": [(10, True), (20, True), (40, True), (50, True)],
        "is_general_rebaseline": [(10, True), (40, True), (50, True)],
        "is_post_relapse_rebaseline": [(20, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (10, 2.5),
            (20, 3),
            (40, 3.5),
            (50, 2),
        ],
        "edss_score_used_as_new_general_reference": [(10, 2.5), (40, 3.5), (50, 2)],
        "is_progression": [(10, True), (40, True)],
        "progression_type": [(10, LABEL_RAW), (40, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(10, 2.5), (40, 3.5)],
        "progression_reference_score": [(10, 1), (40, 2.5)],
        "progression_event_id": [(10, 1), (40, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_5,
        relapse_timestamps=[5, 35],
        targets_dict=test_case_5_targets,
        args_dict={
            "undefined_progression": "end",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 10,
        },
    ), "Test 5 failed!"
    # Test case 6 - followed by PIRA
    test_dataframe_case_6 = pd.DataFrame(
        {
            "days_after_baseline": [0, 20, 40, 60, 80, 100, 120],
            "edss_score": [1, 1.5, 2.5, 2, 3, 3.5, 3.5],
        }
    )
    test_case_6_targets = {
        "days_to_next_relapse": [(0, 15), (20, 5)],
        "days_since_previous_relapse": [
            (20, 5),
            (40, 15),
            (60, 35),
            (80, 55),
            (100, 75),
            (120, 95),
        ],
        "is_post_event_rebaseline": [(40, True), (100, True)],
        "is_raw_pira_rebaseline": [(20, True), (40, True), (100, True)],
        "is_general_rebaseline": [(40, True), (100, True)],
        "is_post_relapse_rebaseline": [(20, True), (40, True)],
        "edss_score_used_as_new_raw_pira_reference": [(20, 1.5), (40, 2.5), (100, 3.5)],
        "edss_score_used_as_new_general_reference": [(40, 2), (100, 3.5)],
        "is_progression": [(40, True), (100, True)],
        "progression_type": [(40, LABEL_UNDEFINED_PROGRESSION), (100, LABEL_PIRA)],
        "progression_score": [(40, 2), (100, 3.5)],
        "progression_reference_score": [(40, 1), (100, 2.5)],
        "progression_event_id": [(40, 1), (100, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_6,
        relapse_timestamps=[15, 25],
        targets_dict=test_case_6_targets,
        args_dict={
            "opt_baseline_type": "fixed",
            "undefined_progression": "end",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 6 failed!"


# -------------------------
# Part 4 - multi-event mode
# --------------------------


def test_multi_event_option():
    # Test case 1 - unconfirmed merged
    test_dataframe_case_1 = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 65, 70, 80, 90],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.5, 4.0],
        }
    )
    test_case_1_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True)],
        "is_raw_pira_rebaseline": [(30, True)],
        "is_general_rebaseline": [(30, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 4.5)],
        "edss_score_used_as_new_general_reference": [(30, 4.5)],
        "is_progression": [(30, True)],
        "progression_type": [(30, LABEL_PIRA)],
        "progression_score": [(30, 4.5)],
        "progression_reference_score": [(30, 1.0)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (65, 1),
            (70, 1),
            (80, 1),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_1,
        relapse_timestamps=[],
        targets_dict=test_case_1_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
        },
    ), "Test 1 failed!"

    # Test case 1b - unconfirmed merged, first only
    test_case_1b_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [],
        "is_raw_pira_rebaseline": [],
        "is_general_rebaseline": [],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [],
        "edss_score_used_as_new_general_reference": [],
        "is_progression": [(30, True)],
        "progression_type": [(30, LABEL_PIRA)],
        "progression_score": [(30, 4.5)],
        "progression_reference_score": [(30, 1.0)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (65, 1),
            (70, 1),
            (80, 1),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_1,
        relapse_timestamps=[],
        targets_dict=test_case_1b_targets,
        args_dict={
            "return_first_event_only": True,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
        },
    ), "Test 1b failed!"

    # Test case 2 - next-confirmed merged
    test_case_2_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True)],
        "is_raw_pira_rebaseline": [(30, True)],
        "is_general_rebaseline": [(30, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 4)],
        "edss_score_used_as_new_general_reference": [(30, 4)],
        "is_progression": [(30, True)],
        "progression_type": [(30, LABEL_PIRA)],
        "progression_score": [(30, 4)],
        "progression_reference_score": [(30, 1.0)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (65, 1),
            (70, 1),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_1,
        relapse_timestamps=[],
        targets_dict=test_case_2_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 2 failed!"

    # Tolerance, case 3
    test_dataframe_case_3 = pd.DataFrame(
        {
            "days_after_baseline": [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                65,
                70,
                80,
                90,
                100,
                110,
            ],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.5, 4.5, 5.0, 5.5],
        }
    )
    test_case_3_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True), (110, True)],
        "is_raw_pira_rebaseline": [(30, True), (80, True), (110, True)],
        "is_general_rebaseline": [(30, True), (80, True), (110, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.5), (80, 4.5), (110, 5.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.5), (80, 4.5), (110, 5.5)],
        "is_progression": [(30, True), (80, True), (110, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA), (110, LABEL_PIRA)],
        "progression_score": [(30, 3.5), (80, 4.5), (110, 5.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.5), (110, 4.5)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2), (110, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_3,
        relapse_timestamps=[],
        targets_dict=test_case_3_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
        },
    ), "Test 3 failed!"

    # Tolerance, case 4
    test_case_4_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (110, True)],
        "is_raw_pira_rebaseline": [(30, True), (110, True)],
        "is_general_rebaseline": [(30, True), (110, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 4.5), (110, 5.5)],
        "edss_score_used_as_new_general_reference": [(30, 4.5), (110, 5.5)],
        "is_progression": [(30, True), (110, True)],
        "progression_type": [(30, LABEL_PIRA), (110, LABEL_PIRA)],
        "progression_score": [(30, 4.5), (110, 5.5)],
        "progression_reference_score": [(30, 1.0), (110, 4.5)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (65, 1),
            (70, 1),
            (80, 1),
            (110, 2),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_3,
        relapse_timestamps=[],
        targets_dict=test_case_4_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
        },
    ), "Test 4 failed!"

    # Tolerance, case 5
    test_case_5_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True)],
        "is_raw_pira_rebaseline": [(30, True)],
        "is_general_rebaseline": [(30, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 5.5)],
        "edss_score_used_as_new_general_reference": [(30, 5.5)],
        "is_progression": [(30, True)],
        "progression_type": [(30, LABEL_PIRA)],
        "progression_score": [(30, 5.5)],
        "progression_reference_score": [(30, 1.0)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (65, 1),
            (70, 1),
            (80, 1),
            (90, 1),
            (100, 1),
            (110, 1),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_3,
        relapse_timestamps=[],
        targets_dict=test_case_5_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
        },
    ), "Test 5 failed!"

    # Test case 6 - dip, unconfirmed
    test_dataframe_case_6 = pd.DataFrame(
        {
            "days_after_baseline": [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                65,
                70,
                80,
                90,
                100,
                110,
            ],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 3.0, 4.0, 4.5, 4.5, 5.0, 5.5],
        }
    )
    test_case_6_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True), (110, True)],
        "is_raw_pira_rebaseline": [(30, True), (80, True), (110, True)],
        "is_general_rebaseline": [(30, True), (80, True), (110, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.5), (80, 4.5), (110, 5.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.5), (80, 4.5), (110, 5.5)],
        "is_progression": [(30, True), (80, True), (110, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA), (110, LABEL_PIRA)],
        "progression_score": [(30, 3.5), (80, 4.5), (110, 5.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.5), (110, 4.5)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2), (110, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_6,
        relapse_timestamps=[],
        targets_dict=test_case_6_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
        },
    ), "Test 6 failed!"

    # Test case 7 - dip, unconfirmed
    test_case_7_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True), (110, True)],
        "is_raw_pira_rebaseline": [(30, True), (80, True), (110, True)],
        "is_general_rebaseline": [(30, True), (80, True), (110, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.5), (80, 4.5), (110, 5.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.5), (80, 4.5), (110, 5.5)],
        "is_progression": [(30, True), (80, True), (110, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA), (110, LABEL_PIRA)],
        "progression_score": [(30, 3.5), (80, 4.5), (110, 5.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.5), (110, 4.5)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2), (110, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_6,
        relapse_timestamps=[],
        targets_dict=test_case_7_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
        },
    ), "Test 7 failed!"

    # Test case 8 - dip, unconfirmed
    test_case_8_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True)],
        "is_raw_pira_rebaseline": [(30, True), (80, True)],
        "is_general_rebaseline": [(30, True), (80, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.5), (80, 5.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.5), (80, 5.5)],
        "is_progression": [(30, True), (80, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(30, 3.5), (80, 5.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.5)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (80, 2),
            (90, 2),
            (100, 2),
            (110, 2),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_6,
        relapse_timestamps=[],
        targets_dict=test_case_8_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
        },
    ), "Test 8 failed!"

    # Test case 9 - dip, next-confirmed
    test_dataframe_case_9 = pd.DataFrame(
        {
            "days_after_baseline": [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                65,
                70,
                80,
                90,
                100,
                110,
            ],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 3.0, 4.0, 4.5, 4.5, 5.0, 5.5],
        }
    )
    test_case_9_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (70, True)],
        "is_raw_pira_rebaseline": [(30, True), (70, True)],
        "is_general_rebaseline": [(30, True), (70, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3), (70, 4.5)],
        "edss_score_used_as_new_general_reference": [(30, 3), (70, 4.5)],
        "is_progression": [(30, True), (70, True)],
        "progression_type": [(30, LABEL_PIRA), (70, LABEL_PIRA)],
        "progression_score": [(30, 3), (70, 4.5)],
        "progression_reference_score": [(30, 1.0), (70, 3.0)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (70, 2), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_9,
        relapse_timestamps=[],
        targets_dict=test_case_9_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 9 failed!"

    # Test case 10 - dip, next-confirmed
    test_case_10_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (70, True)],
        "is_raw_pira_rebaseline": [(30, True), (70, True)],
        "is_general_rebaseline": [(30, True), (70, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.0), (70, 4.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.0), (70, 4.5)],
        "is_progression": [(30, True), (70, True)],
        "progression_type": [(30, LABEL_PIRA), (70, LABEL_PIRA)],
        "progression_score": [(30, 3.0), (70, 4.5)],
        "progression_reference_score": [(30, 1.0), (70, 3.0)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (50, 1),
            (70, 2),
            (80, 2),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_9,
        relapse_timestamps=[],
        targets_dict=test_case_10_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 10 failed!"

    # Test case 11 - dip, next-confirmed
    test_case_11_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (70, True)],
        "is_raw_pira_rebaseline": [(30, True), (70, True)],
        "is_general_rebaseline": [(30, True), (70, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.0), (70, 5.0)],
        "edss_score_used_as_new_general_reference": [(30, 3.0), (70, 5.0)],
        "is_progression": [(30, True), (70, True)],
        "progression_type": [(30, LABEL_PIRA), (70, LABEL_PIRA)],
        "progression_score": [(30, 3.0), (70, 5.0)],
        "progression_reference_score": [(30, 1.0), (70, 3.0)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (50, 1),
            (70, 2),
            (80, 2),
            (90, 2),
            (100, 2),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_9,
        relapse_timestamps=[],
        targets_dict=test_case_11_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 11 failed!"

    # Test case 12 - dip, 20 units confirmed
    test_dataframe_case_12 = pd.DataFrame(
        {
            "days_after_baseline": [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                65,
                70,
                80,
                90,
                100,
                110,
            ],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 3.0, 4.0, 4.5, 4.5, 5.0, 5.5],
        }
    )
    test_case_12_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True)],
        "is_raw_pira_rebaseline": [(30, True), (80, True)],
        "is_general_rebaseline": [(30, True), (80, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.5), (80, 4.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.5), (80, 4.5)],
        "is_progression": [(30, True), (80, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(30, 3.5), (80, 4.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.5)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_12,
        relapse_timestamps=[],
        targets_dict=test_case_12_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
        },
    ), "Test 12 failed!"

    # Test case 13 - dip, 20 units confirmed
    test_case_13_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True)],
        "is_raw_pira_rebaseline": [(30, True), (80, True)],
        "is_general_rebaseline": [(30, True), (80, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.5), (80, 4.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.5), (80, 4.5)],
        "is_progression": [(30, True), (80, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(30, 3.5), (80, 4.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.5)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_12,
        relapse_timestamps=[],
        targets_dict=test_case_13_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
        },
    ), "Test 13 failed!"

    # Test case 14 - dip, 20 units confirmed
    test_case_14_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True)],
        "is_raw_pira_rebaseline": [(30, True), (80, True)],
        "is_general_rebaseline": [(30, True), (80, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.5), (80, 4.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.5), (80, 4.5)],
        "is_progression": [(30, True), (80, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(30, 3.5), (80, 4.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.5)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (80, 2),
        ],  # , (90, 2)
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_12,
        relapse_timestamps=[],
        targets_dict=test_case_14_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
        },
    ), "Test 14 failed!"

    # Test case 15 - dip with stagnation, next-confirmed
    test_dataframe_case_15 = pd.DataFrame(
        {
            "days_after_baseline": [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                65,
                70,
                80,
                90,
                100,
                110,
            ],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 3.0, 3.0, 4.5, 4.5, 5.0, 5.5],
        }
    )
    test_case_15_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True)],
        "is_raw_pira_rebaseline": [(30, True), (80, True)],
        "is_general_rebaseline": [(30, True), (80, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.0), (80, 4.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.0), (80, 4.5)],
        "is_progression": [(30, True), (80, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(30, 3.0), (80, 4.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.0)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_15,
        relapse_timestamps=[],
        targets_dict=test_case_15_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 15 failed!"

    # Test case 16 - dip with stagnation, next-confirmed
    test_case_16_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True)],
        "is_raw_pira_rebaseline": [(30, True), (80, True)],
        "is_general_rebaseline": [(30, True), (80, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.0), (80, 4.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.0), (80, 4.5)],
        "is_progression": [(30, True), (80, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(30, 3.0), (80, 4.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.0)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_15,
        relapse_timestamps=[],
        targets_dict=test_case_16_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 16 failed!"

    # Test case 17 - dip with stagnation, next-confirmed
    test_case_17_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True)],
        "is_raw_pira_rebaseline": [(30, True), (80, True)],
        "is_general_rebaseline": [(30, True), (80, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.0), (80, 5.0)],
        "edss_score_used_as_new_general_reference": [(30, 3.0), (80, 5.0)],
        "is_progression": [(30, True), (80, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(30, 3.0), (80, 5.0)],
        "progression_reference_score": [(30, 1.0), (80, 3.0)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (50, 1),
            (80, 2),
            (90, 2),
            (100, 2),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_15,
        relapse_timestamps=[],
        targets_dict=test_case_17_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 17 failed!"

    # Test case 18 - dip with stagnation, 20 units last confirmed
    test_dataframe_case_18 = pd.DataFrame(
        {
            "days_after_baseline": [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                65,
                70,
                80,
                90,
                100,
                110,
            ],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 3.0, 3.0, 4.5, 4.5, 5.0, 5.5],
        }
    )
    test_case_18_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True)],
        "is_raw_pira_rebaseline": [(30, True), (65, True), (80, True)],
        "is_general_rebaseline": [(30, True), (65, True), (80, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.5), (65, 3.0), (80, 4.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.5), (65, 3.0), (80, 4.5)],
        "is_progression": [(30, True), (80, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(30, 3.5), (80, 4.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.0)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_18,
        relapse_timestamps=[],
        targets_dict=test_case_18_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
        },
    ), "Test 18 failed!"

    # Test case 19 - dip with stagnation, 20 units last confirmed
    test_case_19_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True)],
        "is_raw_pira_rebaseline": [(30, True), (65, True), (80, True)],
        "is_general_rebaseline": [(30, True), (65, True), (80, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.5), (65, 3.0), (80, 4.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.5), (65, 3.0), (80, 4.5)],
        "is_progression": [(30, True), (80, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(30, 3.5), (80, 4.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.0)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_18,
        relapse_timestamps=[],
        targets_dict=test_case_19_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
        },
    ), "Test 19 failed!"

    # Test case 20 - dip with stagnation, 20 units last confirmed
    test_case_20_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (80, True)],
        "is_raw_pira_rebaseline": [(30, True), (65, True), (80, True)],
        "is_general_rebaseline": [(30, True), (65, True), (80, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [(30, 3.5), (65, 3.0), (80, 4.5)],
        "edss_score_used_as_new_general_reference": [(30, 3.5), (65, 3.0), (80, 4.5)],
        "is_progression": [(30, True), (80, True)],
        "progression_type": [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        "progression_score": [(30, 3.5), (80, 4.5)],
        "progression_reference_score": [(30, 1.0), (80, 3.0)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_18,
        relapse_timestamps=[],
        targets_dict=test_case_20_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
        },
    ), "Test 20 failed!"

    # Test case 21 - undefined must remain singular
    test_dataframe_case_21 = pd.DataFrame(
        {"days_after_baseline": [0, 8, 36, 48, 60], "edss_score": [1, 1.5, 2.5, 3, 3.5]}
    )

    test_case_21_targets = {
        "days_to_next_relapse": [(0, 6), (8, 22)],
        "days_since_previous_relapse": [(8, 2), (36, 6), (48, 18), (60, 30)],
        "is_post_event_rebaseline": [(36, True), (60, True)],
        "is_raw_pira_rebaseline": [(36, True), (60, True)],
        "is_general_rebaseline": [(36, True), (60, True)],
        "is_post_relapse_rebaseline": [(36, True)],
        "edss_score_used_as_new_raw_pira_reference": [(36, 2.5), (60, 3.5)],
        "edss_score_used_as_new_general_reference": [(36, 2.5), (60, 3.5)],
        "is_progression": [(36, True), (60, True)],
        "progression_type": [(36, LABEL_UNDEFINED_PROGRESSION), (60, LABEL_PIRA)],
        "progression_score": [(36, 2.5), (60, 3.5)],
        "progression_reference_score": [(36, 1.0), (60, 2.5)],
        "progression_event_id": [(36, 1), (60, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_21,
        relapse_timestamps=[6, 30],
        targets_dict=test_case_21_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_days": 4,
            "opt_raw_after_relapse_max_days": 4,
        },
    ), "Test 21 failed!"

    # Test case 22 - with relapse, unconfirmed
    test_dataframe_case_22 = pd.DataFrame(
        {
            "days_after_baseline": [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
            ],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
        }
    )
    test_case_22_targets = {
        "days_to_next_relapse": [
            (0, 55),
            (10, 45),
            (20, 35),
            (30, 25),
            (40, 15),
            (50, 5),
        ],
        "days_since_previous_relapse": [
            (60, 5),
            (70, 15),
            (80, 25),
            (90, 35),
            (100, 45),
            (110, 55),
            (120, 65),
        ],
        "is_post_event_rebaseline": [(30, True), (60, True), (100, True)],
        "is_raw_pira_rebaseline": [(30, True), (60, True), (80, True), (100, True)],
        "is_general_rebaseline": [(30, True), (60, True), (100, True)],
        "is_post_relapse_rebaseline": [(80, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (30, 2.5),
            (60, 4.0),
            (80, 4.5),
            (100, 6.5),
        ],
        "edss_score_used_as_new_general_reference": [(30, 2.5), (60, 4.0), (100, 6.5)],
        "is_progression": [(30, True), (60, True), (100, True)],
        "progression_type": [(30, LABEL_PIRA), (60, LABEL_RAW), (100, LABEL_PIRA)],
        "progression_score": [(30, 2.5), (60, 4.0), (100, 6.5)],
        "progression_reference_score": [(30, 1.0), (60, 2.5), (100, 4.5)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (100, 3),
            (110, 3),
            (120, 3),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_22,
        relapse_timestamps=[55],
        targets_dict=test_case_22_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 15,
        },
    ), "Test 22 failed!"

    # Test case 23 - with relapse, next-confirmed
    test_case_23_targets = {
        "days_to_next_relapse": [
            (0, 55),
            (10, 45),
            (20, 35),
            (30, 25),
            (40, 15),
            (50, 5),
        ],
        "days_since_previous_relapse": [
            (60, 5),
            (70, 15),
            (80, 25),
            (90, 35),
            (100, 45),
            (110, 55),
            (120, 65),
        ],
        "is_post_event_rebaseline": [(30, True), (50, True), (100, True)],
        "is_raw_pira_rebaseline": [(30, True), (50, True), (80, True), (100, True)],
        "is_general_rebaseline": [(30, True), (50, True), (100, True)],
        "is_post_relapse_rebaseline": [(80, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (30, 2.0),
            (50, 4.0),
            (80, 4.5),
            (100, 6.0),
        ],
        "edss_score_used_as_new_general_reference": [(30, 2.0), (50, 4.0), (100, 6.0)],
        "is_progression": [(30, True), (50, True), (100, True)],
        "progression_type": [(30, LABEL_PIRA), (50, LABEL_RAW), (100, LABEL_PIRA)],
        "progression_score": [(30, 2.0), (50, 4.0), (100, 6.0)],
        "progression_reference_score": [(30, 1.0), (50, 2.0), (100, 4.5)],
        "progression_event_id": [
            (30, 1),
            (50, 2),
            (60, 2),
            (70, 2),
            (100, 3),
            (110, 3),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_22,
        relapse_timestamps=[55],
        targets_dict=test_case_23_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 15,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 23 failed!"

    # Test case 24 - with relapse, unconfirmed
    test_dataframe_case_24 = pd.DataFrame(
        {
            "days_after_baseline": [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
            ],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
        }
    )
    test_case_24_targets = {
        "days_to_next_relapse": [
            (0, 65),
            (10, 55),
            (20, 45),
            (30, 35),
            (40, 25),
            (50, 15),
            (60, 5),
        ],
        "days_since_previous_relapse": [
            (70, 5),
            (80, 15),
            (90, 25),
            (100, 35),
            (110, 45),
            (120, 55),
        ],
        "is_post_event_rebaseline": [(30, True), (70, True), (110, True)],
        "is_raw_pira_rebaseline": [(30, True), (70, True), (90, True), (110, True)],
        "is_general_rebaseline": [(30, True), (70, True), (110, True)],
        "is_post_relapse_rebaseline": [(90, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (30, 3.0),
            (70, 4.5),
            (90, 5.0),
            (110, 6.5),
        ],
        "edss_score_used_as_new_general_reference": [(30, 3.0), (70, 4.5), (110, 6.5)],
        "is_progression": [(30, True), (70, True), (110, True)],
        "progression_type": [(30, LABEL_PIRA), (70, LABEL_RAW), (110, LABEL_PIRA)],
        "progression_score": [(30, 3.0), (70, 4.5), (110, 6.5)],
        "progression_reference_score": [(30, 1.0), (70, 3.0), (110, 5.0)],
        "progression_event_id": [
            (30, 1),
            (40, 1),
            (50, 1),
            (70, 2),
            (80, 2),
            (110, 3),
            (120, 3),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_24,
        relapse_timestamps=[65],
        targets_dict=test_case_24_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 15,
        },
    ), "Test 24 failed!"

    # Test case 25 - with relapse, unconfirmed
    test_case_25_targets = {
        "days_to_next_relapse": [
            (0, 65),
            (10, 55),
            (20, 45),
            (30, 35),
            (40, 25),
            (50, 15),
            (60, 5),
        ],
        "days_since_previous_relapse": [
            (70, 5),
            (80, 15),
            (90, 25),
            (100, 35),
            (110, 45),
            (120, 55),
        ],
        "is_post_event_rebaseline": [(30, True), (60, True), (110, True)],
        "is_raw_pira_rebaseline": [(30, True), (60, True), (90, True), (110, True)],
        "is_general_rebaseline": [(30, True), (60, True), (110, True)],
        "is_post_relapse_rebaseline": [(90, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (30, 2.5),
            (60, 4.5),
            (90, 5.0),
            (110, 6.0),
        ],
        "edss_score_used_as_new_general_reference": [(30, 2.5), (60, 4.5), (110, 6.0)],
        "is_progression": [(30, True), (60, True), (110, True)],
        "progression_type": [(30, LABEL_PIRA), (60, LABEL_RAW), (110, LABEL_PIRA)],
        "progression_score": [(30, 2.5), (60, 4.5), (110, 6.0)],
        "progression_reference_score": [(30, 1.0), (60, 2.5), (110, 5.0)],
        "progression_event_id": [(30, 1), (40, 1), (60, 2), (70, 2), (80, 2), (110, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_24,
        relapse_timestamps=[65],
        targets_dict=test_case_25_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 15,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 25 failed!"

    # Test case 26 - with 'end'
    test_dataframe_case_26 = pd.DataFrame(
        {
            "days_after_baseline": [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                140,
                150,
                160,
                170,
            ],
            "edss_score": [0, 1, 2, 1.5, 1.5, 2, 2, 1.5, 1.5, 2.5, 2, 2, 2.5, 2.5],
        }
    )
    test_case_26_targets = {
        "days_to_next_relapse": [
            (0, 15),
            (10, 5),
            (20, 65),
            (30, 55),
            (40, 45),
            (50, 35),
            (60, 25),
            (70, 15),
            (80, 5),
        ],
        "days_since_previous_relapse": [
            (20, 5),
            (30, 15),
            (40, 25),
            (50, 35),
            (60, 45),
            (70, 55),
            (80, 65),
            (90, 5),
            (140, 55),
            (150, 65),
            (160, 75),
            (170, 85),
        ],
        "is_post_event_rebaseline": [(10, True), (160, True)],
        "is_raw_pira_rebaseline": [(10, True), (140, True), (160, True)],
        "is_general_rebaseline": [(10, True), (160, True)],
        "is_post_relapse_rebaseline": [(140, True)],
        "edss_score_used_as_new_raw_pira_reference": [
            (10, 1.5),
            (140, 2.0),
            (160, 2.5),
        ],
        "edss_score_used_as_new_general_reference": [(10, 1.5), (160, 2.5)],
        "is_progression": [(10, True), (160, True)],
        "progression_type": [(10, LABEL_RAW), (160, LABEL_UNDEFINED_PROGRESSION)],
        "progression_score": [(10, 1.5), (160, 2.5)],
        "progression_reference_score": [(10, 0), (160, 1.5)],
        "progression_event_id": [(10, 1), (20, 1), (160, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_26,
        relapse_timestamps=[15, 85],
        targets_dict=test_case_26_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "undefined_progression": "end",
            "opt_raw_before_relapse_max_days": 5,
            "opt_raw_after_relapse_max_days": 15,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_larger_increment_from_0": False,
        },
    ), "Test 26 failed!"

    # Test case 27 - stagnation then increase, unconfirmed
    test_dataframe_case_27 = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 65, 70],
            "edss_score": [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0],
        }
    )
    test_case_27_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True)],
        "is_raw_pira_rebaseline": [(30, True)],
        "is_general_rebaseline": [(30, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [
            (30, 4.0),
        ],
        "edss_score_used_as_new_general_reference": [(30, 4.0)],
        "is_progression": [(30, True)],
        "progression_type": [(30, LABEL_PIRA)],
        "progression_score": [(30, 4.0)],
        "progression_reference_score": [(30, 1.0)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (60, 1), (65, 1), (70, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_27,
        relapse_timestamps=[],
        targets_dict=test_case_27_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "undefined_progression": "re-baselining only",
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 27 failed!"

    # Test case 28 - stagnation then increase, next-confirmed
    test_case_28_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True)],
        "is_raw_pira_rebaseline": [(30, True)],
        "is_general_rebaseline": [(30, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [
            (30, 3.5),
        ],
        "edss_score_used_as_new_general_reference": [(30, 3.5)],
        "is_progression": [(30, True)],
        "progression_type": [(30, LABEL_PIRA)],
        "progression_score": [(30, 3.5)],
        "progression_reference_score": [(30, 1.0)],
        "progression_event_id": [(30, 1), (40, 1), (50, 1), (60, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_27,
        relapse_timestamps=[],
        targets_dict=test_case_28_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "undefined_progression": "re-baselining only",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 28 failed!"

    # Test case 29 - maximal distance settings
    test_dataframe_case_29 = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 50, 60, 65, 70],  # , 40
            "edss_score": [1, 1, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5],  # , 2.5
        }
    )
    test_case_29_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (50, True), (65, True)],
        "is_raw_pira_rebaseline": [(30, True), (50, True), (65, True)],
        "is_general_rebaseline": [(30, True), (50, True), (65, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [
            (30, 2.0),
            (50, 3.0),
            (65, 4.5),
        ],
        "edss_score_used_as_new_general_reference": [
            (30, 2.0),
            (50, 3.0),
            (65, 4.5),
        ],
        "is_progression": [(30, True), (50, True), (65, True)],
        "progression_type": [
            (30, LABEL_PIRA),
            (50, LABEL_PIRA),
            (65, LABEL_PIRA),
        ],
        "progression_score": [
            (30, 2.0),
            (50, 3.0),
            (65, 4.5),
        ],
        "progression_reference_score": [
            (30, 1.0),
            (50, 2.0),
            (65, 3.0),
        ],
        "progression_event_id": [(30, 1), (50, 2), (65, 3), (70, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_29,
        relapse_timestamps=[],
        targets_dict=test_case_29_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "continuous_events_max_merge_distance": 9.9,
            "opt_require_confirmation": False,
        },
    ), "Test 29 failed!"

    # Test case 30 - maximal distance settings
    test_case_30_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True), (50, True)],
        "is_raw_pira_rebaseline": [(30, True), (50, True)],
        "is_general_rebaseline": [(30, True), (50, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [
            (30, 2.0),
            (50, 4.5),
        ],
        "edss_score_used_as_new_general_reference": [
            (30, 2.0),
            (50, 4.5),
        ],
        "is_progression": [(30, True), (50, True)],
        "progression_type": [
            (30, LABEL_PIRA),
            (50, LABEL_PIRA),
        ],
        "progression_score": [
            (30, 2.0),
            (50, 4.5),
        ],
        "progression_reference_score": [
            (30, 1.0),
            (50, 2.0),
        ],
        "progression_event_id": [(30, 1), (50, 2), (60, 2), (65, 2), (70, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_29,
        relapse_timestamps=[],
        targets_dict=test_case_30_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "continuous_events_max_merge_distance": 10,
            "opt_require_confirmation": False,
        },
    ), "Test 30 failed!"

    # Test case 31 - maximal distance settings
    test_case_31_targets = {
        "days_to_next_relapse": [],
        "days_since_previous_relapse": [],
        "is_post_event_rebaseline": [(30, True)],
        "is_raw_pira_rebaseline": [(30, True)],
        "is_general_rebaseline": [(30, True)],
        "is_post_relapse_rebaseline": [],
        "edss_score_used_as_new_raw_pira_reference": [
            (30, 4.5),
        ],
        "edss_score_used_as_new_general_reference": [
            (30, 4.5),
        ],
        "is_progression": [(30, True)],
        "progression_type": [
            (30, LABEL_PIRA),
        ],
        "progression_score": [
            (30, 4.5),
        ],
        "progression_reference_score": [(30, 1.0)],
        "progression_event_id": [(30, 1), (50, 1), (60, 1), (65, 1), (70, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_29,
        relapse_timestamps=[],
        targets_dict=test_case_31_targets,
        args_dict={
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 31 failed!"


if __name__ == "__main__":
    print("\nPart 1 - building blocks\n")
    print("Testing 'is_above_progress_threshold'...")
    test_is_above_progress_threshold()

    print("Testing 'get_confirmation_scores_dataframe'...")
    test_get_confirmation_scores_dataframe()

    print("Testing 'check_confirmation_scores_and_get_confirmed_score'...")
    test_check_confirmation_scores_and_get_confirmed_score()

    print("Testing 'backtrack_minimal_distance_compatible_reference'...")
    test_backtrack_minimal_distance_compatible_reference()

    print("\nPart 2 - relapse independent progression\n")
    print("Testing confirmation...")
    test_relapse_independent_confirmation()

    print("Testing baselines...")
    test_relapse_independent_baselines()
    test_roving_baseline_with_minimal_score()

    print("Testing minimal distance...")
    test_relapse_independent_minimal_distance()

    print("Testing first vs. all events...")
    test_relapse_independent_first_vs_all_events()

    print("Testing multiple events re-baselining...")
    test_relapse_independent_multiple_events_rebaselining()

    print("\nPart 3 - progression with relapses\n")
    print("Testing function to add relapses to a follow-up...")
    test_add_relapses_to_follow_up()

    print("Testing function to get post-relapse re-baselining timestamps...")
    test_get_post_relapse_rebaseline_timestamps()

    print("Testing independence of baselines for roving...")
    test_roving_raw_pira_descends_to_general()

    print("Testing RAW/PIRA label assignment for unconfirmed progression...")
    test_raw_pira_unconfirmed()

    print("Testing RAW/PIRA label assignment for confirmed progression...")
    test_raw_pira_confirmed()
    test_raw_pira_confirmed_mueller_2025()

    print("Testing post-relapse re-baselining depending on post-relapse score...")
    test_post_relapse_rebaselining_higher_equal_lower()

    print("Testing post-relapse re-baselining with minimal distance backtracking...")
    test_post_relapse_rebaselining_backtracking()

    print(
        "Testing post-relapse re-baselining with missing assessments and overlapping relapses..."
    )
    test_post_relapse_rebaselining_with_missing_and_overlapping()

    print("Testing post-relapse re-baselining with event at the same assessment...")
    test_post_relapse_rebaselining_with_event()

    print("Testing option to never annotate undefined events...")
    test_undefined_progression_never_option()

    print("Testing option to always annotate undefined events...")
    test_undefined_progression_all_option()

    print("Testing option to annotate undefined events if they don't mask RAW/PIRA...")
    test_undefined_progression_end_option()

    print("\nPart 4 - multi-event mode\n")
    print("Testing multi-event mode...")
    test_multi_event_option()

    print("\nAll tests successfully completed.\n")
