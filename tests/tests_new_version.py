"""Test edssannotation.py functionality.

Execute

    python -m tests.tests_new_version

in parent to run tests!

This is a bit messy for now. Use tools.visualization.py for some
visual testing.

"""

import numpy as np
import pandas as pd

from collections import Counter

from definitions import edssannotation


# Test the increase/decrease delta check
def test_is_large_enough_increase_or_decrease():
    # Test 1 - minimum required increase + 0.5 irrespective of reference
    test_cases_1 = [
        {"current": 0, "reference": 0, "target": [False, False]},
        # Increase
        {"current": 0.5, "reference": 0, "target": [True, False]},
        {"current": 1.0, "reference": 0, "target": [True, False]},
        {"current": 1.5, "reference": 0, "target": [True, False]},
        {"current": 2.0, "reference": 2.0, "target": [False, False]},
        {"current": 2.5, "reference": 2.0, "target": [True, False]},
        {"current": 3.0, "reference": 2.0, "target": [True, False]},
        # Decrease
        {"current": 0, "reference": 0.5, "target": [False, True]},
        {"current": 0, "reference": 1.0, "target": [False, True]},
        {"current": 0, "reference": 1.5, "target": [False, True]},
        {"current": 2.0, "reference": 2.0, "target": [False, False]},
        {"current": 9.5, "reference": 10.0, "target": [False, True]},
        {"current": 5.5, "reference": 6.0, "target": [False, True]},
        {"current": 5.0, "reference": 6.0, "target": [False, True]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    opt_max_score_that_requires_plus_1=-1,
                    opt_larger_increment_from_0=False,
                )._is_large_enough_increase_or_decrease(
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
        {"current": 0, "reference": 0, "target": [False, False]},
        # Increase
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [True, False]},
        {"current": 1.5, "reference": 0, "target": [True, False]},
        {"current": 2.0, "reference": 2.0, "target": [False, False]},
        {"current": 2.5, "reference": 2.0, "target": [False, False]},
        {"current": 3.0, "reference": 2.0, "target": [True, False]},
        {"current": 10.0, "reference": 9.5, "target": [False, False]},
        # Decrease
        {"current": 0, "reference": 0.5, "target": [False, False]},
        {"current": 0, "reference": 1.0, "target": [False, True]},
        {"current": 0, "reference": 1.5, "target": [False, True]},
        {"current": 9.5, "reference": 10.0, "target": [False, False]},
        {"current": 5.5, "reference": 6.0, "target": [False, False]},
        {"current": 5.0, "reference": 6.0, "target": [False, True]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    opt_max_score_that_requires_plus_1=10.0,
                    opt_larger_increment_from_0=False,
                )._is_large_enough_increase_or_decrease(
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
        {"current": 0, "reference": 0, "target": [False, False]},
        # Increase
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [False, False]},
        {"current": 1.5, "reference": 0, "target": [True, False]},
        {"current": 2.0, "reference": 2.0, "target": [False, False]},
        {"current": 2.5, "reference": 2.0, "target": [True, False]},
        {"current": 3.0, "reference": 2.0, "target": [True, False]},
        # Decrease
        {"current": 0, "reference": 0.5, "target": [False, False]},
        {"current": 0, "reference": 1.0, "target": [False, False]},
        {"current": 0, "reference": 1.5, "target": [False, True]},
        {"current": 2.0, "reference": 2.0, "target": [False, False]},
        {"current": 2.0, "reference": 2.5, "target": [False, True]},
        {"current": 2.0, "reference": 3.0, "target": [False, True]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    opt_max_score_that_requires_plus_1=1,
                    opt_larger_increment_from_0=True,
                )._is_large_enough_increase_or_decrease(
                    current_edss=test_case["current"],
                    reference_edss=test_case["reference"],
                )
                for test_case in test_cases_3
            ]
        ),
        np.array([test_case["target"] for test_case in test_cases_3]),
        err_msg="Minimal increase + 1.5 from 0 failed!",
    )
    # Test 4 - standard case
    test_cases_4 = [
        {"current": 0, "reference": 0, "target": [False, False]},
        # Increase
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [False, False]},
        {"current": 1.5, "reference": 0, "target": [True, False]},
        {"current": 2.0, "reference": 2.0, "target": [False, False]},
        {"current": 2.5, "reference": 2.0, "target": [False, False]},
        {"current": 3.0, "reference": 2.0, "target": [True, False]},
        {"current": 5.5, "reference": 5.0, "target": [False, False]},
        {"current": 6.0, "reference": 5.0, "target": [True, False]},
        {"current": 6.0, "reference": 5.5, "target": [True, False]},
        {"current": 6.5, "reference": 5.5, "target": [True, False]},
        # Decrease
        {"current": 0, "reference": 0.5, "target": [False, False]},
        {"current": 0, "reference": 1.0, "target": [False, False]},
        {"current": 0, "reference": 1.5, "target": [False, True]},
        {"current": 9.5, "reference": 10.0, "target": [False, True]},
        {"current": 5.5, "reference": 6.0, "target": [False, True]},
        {"current": 5.0, "reference": 6.0, "target": [False, True]},
        {"current": 5.0, "reference": 5.5, "target": [False, False]},
        {"current": 4.5, "reference": 5.5, "target": [False, True]},
        {"current": 4.5, "reference": 5.0, "target": [False, False]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation()._is_large_enough_increase_or_decrease(
                    current_edss=test_case["current"],
                    reference_edss=test_case["reference"],
                )
                for test_case in test_cases_4
            ]
        ),
        np.array([test_case["target"] for test_case in test_cases_4]),
        err_msg="Standard minimal increase settings failed!",
    )


# Test the confirmation dataframe extraction
def test_get_confirmation_scores_dataframe():
    test_dataframe = pd.DataFrame(
        {"timestamp": [0, 10, 20, 30, 40, 50], "score": [0, 1, 2, 3, 4, 5]}
    )
    # Test case 1 - sustained, assessments available
    # Must yield all assessments following the first one.
    test_case_1 = edssannotation.EDSSAnnotation(
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
    test_case_2 = edssannotation.EDSSAnnotation(
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
    assert test_case_2.equals(test_dataframe[test_dataframe["timestamp"] > 50]), (
        "Test 2 failed!"
    )
    # Test case 3 - sustained, minimal time interval
    # Must yield all assessments following the second one.
    test_case_3 = edssannotation.EDSSAnnotation(
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
    test_case_4 = edssannotation.EDSSAnnotation(
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
    assert test_case_4.equals(test_dataframe[test_dataframe["timestamp"] > 50]), (
        "Test 4 failed!"
    )
    # Test case 5 - time interval, assessments available
    # Must yield all assessments following the first one
    # until and including the assessment at 40.
    test_case_5 = edssannotation.EDSSAnnotation(
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
    test_case_6 = edssannotation.EDSSAnnotation(
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
    assert test_case_6.equals(test_dataframe[test_dataframe["timestamp"] > 50]), (
        "Test 6 failed!"
    )
    # Test case 7 - time interval, right side constrained
    # Next is further away, but within tolerance.
    test_case_7 = edssannotation.EDSSAnnotation(
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
    test_case_8 = edssannotation.EDSSAnnotation(
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
    assert test_case_8.equals(test_dataframe[test_dataframe["timestamp"] > 50]), (
        "Test 8 failed!"
    )
    # Test case 9 - time interval, left side tolerance standard
    # Must yield assesments at 20 and 30 (30 is first).
    test_case_9 = edssannotation.EDSSAnnotation(
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
    test_case_10 = edssannotation.EDSSAnnotation(
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
    test_case_11 = edssannotation.EDSSAnnotation(
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
    test_case_12 = edssannotation.EDSSAnnotation(
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
    test_case_13 = edssannotation.EDSSAnnotation(
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
    # Part 1 - increase
    # Note that the timestamps are not required anymore!
    # TODO: ADD TESTS FOR ADDITIONAL LOWER THRESHOLD!
    test_dataframe = pd.DataFrame({"score": [5, 4.5, 5]})
    # Test case 1 - minimum, against a reference of 4
    # Not confirmed, confirmed score is nan.
    assert edssannotation.EDSSAnnotation(
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
        False,
        np.nan,
    ), "Test 1 failed!"
    # Test case 2 - minimum, against a reference of 3.5
    # Confirmed, confirmed score is 3.5.
    assert edssannotation.EDSSAnnotation(
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
        False,
        4.5,
    ), "Test 2 failed!"
    # Test case 3 - monotonic, against a reference of 3.5
    # Not confirmed, confirmed score is nan
    assert edssannotation.EDSSAnnotation(
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
        False,
        np.nan,
    ), "Test 3 failed!"

    # Test case 4 - minimum, one score
    # Confirmed, confirmed score is 5.
    assert edssannotation.EDSSAnnotation(
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
        False,
        5,
    ), "Test 4 failed!"
    # Test case 5 - no scores available
    # Not confirmed, confirmed score is nan.
    assert edssannotation.EDSSAnnotation(
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
        False,
        np.nan,
    ), "Test 5 failed!"

    # Part 2 - decrease
    test_dataframe = pd.DataFrame({"score": [4, 4.5, 4]})
    # Test case 1 - minimum, against a reference of 5
    # Not confirmed, confirmed score is nan.
    assert edssannotation.EDSSAnnotation(
        opt_confirmation_type="minimum",
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
        edss_score_column_name="score",
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=4,
        current_reference=5,
        confirmation_scores_dataframe=test_dataframe,
        additional_lower_threshold=0,
    ) == (
        False,
        False,
        np.nan,
    ), "Test 6 failed!"
    # Test case 2 - minimum, against a reference of 5.5
    # Confirmed, confirmed score is 4.5.
    assert edssannotation.EDSSAnnotation(
        opt_confirmation_type="minimum",
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
        edss_score_column_name="score",
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=4,
        current_reference=5.5,
        confirmation_scores_dataframe=test_dataframe,
        additional_lower_threshold=0,
    ) == (
        False,
        True,
        4.5,
    ), "Test 7 failed!"
    # Test case 3 - monotonic, against a reference of 5.5
    # Not confirmed, confirmed score is nan
    assert edssannotation.EDSSAnnotation(
        opt_confirmation_type="monotonic",
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
        edss_score_column_name="score",
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=4,
        current_reference=5.5,
        confirmation_scores_dataframe=test_dataframe,
        additional_lower_threshold=0,
    ) == (
        False,
        False,
        np.nan,
    ), "Test 8 failed!"

    # Test case 4 - minimum, one score
    # Confirmed, confirmed score is 5.
    assert edssannotation.EDSSAnnotation(
        opt_confirmation_type="monotonic",
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
        edss_score_column_name="score",
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=4,
        current_reference=5,
        confirmation_scores_dataframe=pd.DataFrame({"score": [4]}),
        additional_lower_threshold=0,
    ) == (
        False,
        True,
        4,
    ), "Test 9 failed!"
    # Test case 5 - no scores available
    # Not confirmed, confirmed score is nan.
    assert edssannotation.EDSSAnnotation(
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
        False,
        np.nan,
    ), "Test 10 failed!"


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
    assert edssannotation.EDSSAnnotation(
        opt_minimal_distance_time=20,
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
    )._backtrack_minimal_distance_compatible_reference(
        current_edss=6,
        current_timestamp=60,
        check_increase=True,
        check_decrease=False,
        baselines_df=test_dataframe,
    ) == (
        5.0,
        40,
    ), "Test 1 failed!"
    # Test case 2 - minimal distance is larger than that to the
    # previous reference. No reference score far enough away is
    # low enough to serve as a progression reference.
    assert edssannotation.EDSSAnnotation(
        opt_minimal_distance_time=20,
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
    )._backtrack_minimal_distance_compatible_reference(
        current_edss=5.5,
        current_timestamp=60,
        check_increase=True,
        check_decrease=False,
        baselines_df=test_dataframe,
    ) == (
        np.nan,
        np.nan,
    ), "Test 2 failed!"
    # TODO: Write more tests, write tests for decrease
    # Increase, with increase flag
    bktr_test_3_output = np.array(
        [
            edssannotation.EDSSAnnotation(
                opt_minimal_distance_time=dist
            )._backtrack_minimal_distance_compatible_reference(
                current_edss=4.0,
                current_timestamp=30,
                check_increase=True,
                check_decrease=False,
                baselines_df=pd.DataFrame(
                    {
                        "baseline_timestamp": [0, 10, 20],
                        "baseline_score": [3.5, 3.0, 2.5],
                    }
                ),
            )
            for dist in [0, 9, 10, 11, 19, 20, 21, 29, 30]
        ]
    )
    np.testing.assert_array_equal(
        bktr_test_3_output,
        np.array(
            [
                [2.5, 20],
                [2.5, 20],
                [2.5, 20],
                [3, 10],
                [3, 10],
                [3, 10],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        ),
        err_msg="Test 3 failed!",
    )
    # Increase, with decrease flag
    bktr_test_4_output = np.array(
        [
            edssannotation.EDSSAnnotation(
                opt_minimal_distance_time=dist
            )._backtrack_minimal_distance_compatible_reference(
                current_edss=4.0,
                current_timestamp=30,
                check_increase=False,
                check_decrease=True,
                baselines_df=pd.DataFrame(
                    {
                        "baseline_timestamp": [0, 10, 20],
                        "baseline_score": [3.5, 3.0, 2.5],
                    }
                ),
            )
            for dist in [0, 9, 10, 11, 19, 20, 21, 29, 30]
        ]
    )
    np.testing.assert_array_equal(
        bktr_test_4_output,
        np.array([[np.nan, np.nan] for _ in range(9)]),
        err_msg="Test 4 failed!",
    )
    # Decrease, with decrease flag
    bktr_test_5_output = np.array(
        [
            edssannotation.EDSSAnnotation(
                opt_minimal_distance_time=tmz
            )._backtrack_minimal_distance_compatible_reference(
                current_edss=4.0,
                current_timestamp=30,
                check_increase=False,
                check_decrease=True,
                baselines_df=pd.DataFrame(
                    {
                        "baseline_timestamp": [0, 10, 20],
                        "baseline_score": [4.5, 5.0, 5.5],
                    }
                ),
            )
            for tmz in [0, 9, 10, 11, 19, 20, 21, 29, 30]
        ]
    )
    np.testing.assert_array_equal(
        bktr_test_5_output,
        np.array(
            [
                [5.5, 20],
                [5.5, 20],
                [5.5, 20],
                [5, 10],
                [5, 10],
                [5, 10],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        ),
        err_msg="Test 5 failed!",
    )
    # Decrease, with increase flag
    bktr_test_6_output = np.array(
        [
            edssannotation.EDSSAnnotation(
                opt_minimal_distance_time=tmz
            )._backtrack_minimal_distance_compatible_reference(
                current_edss=4.0,
                current_timestamp=30,
                check_increase=True,
                check_decrease=False,
                baselines_df=pd.DataFrame(
                    {
                        "baseline_timestamp": [0, 10, 20],
                        "baseline_score": [4.5, 5.0, 5.5],
                    }
                ),
            )
            for tmz in [0, 9, 10, 11, 19, 20, 21, 29, 30]
        ]
    )
    np.testing.assert_array_equal(
        bktr_test_6_output,
        np.array([[np.nan, np.nan] for _ in range(9)]),
        err_msg="Test 6 failed!",
    )


if __name__ == "__main__":
    print("\nPart 1 - building blocks\n")
    print("Testing 'is_above_progress_threshold'...")
    test_is_large_enough_increase_or_decrease()

    print("Testing 'get_confirmation_scores_dataframe'...")
    test_get_confirmation_scores_dataframe()

    print("Testing 'check_confirmation_scores_and_get_confirmed_score'...")
    test_check_confirmation_scores_and_get_confirmed_score()

    print("Testing 'backtrack_minimal_distance_compatible_reference'...")
    test_backtrack_minimal_distance_compatible_reference()

    print("\nAll tests successfully completed.\n")
