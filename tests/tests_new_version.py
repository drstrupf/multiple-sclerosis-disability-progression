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


LABEL_PIRA = "PIRA"
LABEL_IMPROVEMENT = "Improvement"


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


def test_check_assessment_for_progression():
    # Returns is_event, is_accrual, is_improvement,
    # event_type, confirmed_event_score,
    # current_baseline_score

    # Without minimal distance, without confirmation
    example_follow_up_1 = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30],
            "edss_score": [4.5, 5.5, 5.5, 4.0],
        }
    )
    example_baselines_1 = example_follow_up_1.iloc[:-1].rename(
        columns={
            "days_after_baseline": "baseline_timestamp",
            "edss_score": "baseline_score",
        }
    )
    test_1_target = pd.DataFrame(
        [
            [True, True, False, LABEL_PIRA, 5.5, 4.5],
            [False, False, False, None, np.nan, 5.5],
            [True, False, True, LABEL_IMPROVEMENT, 4.0, 5.5],
        ]
    )
    test_1_result = pd.DataFrame(
        [
            edssannotation.EDSSAnnotation(
                opt_minimal_distance_time=0,
                opt_require_confirmation=False,
                annotation_mode="experimental-symmetric",
                opt_baseline_type="fixed",
            )._check_assessment_for_progression(
                annotated_df=example_follow_up_1,
                baselines_df=example_baselines_1.iloc[:i],
                current_assessment_index=i,
                additional_lower_threshold=0,
            )
            for i in range(1, 4)
        ]
    )
    assert test_1_result.equals(test_1_target), "Test 1 failed!"

    # With different minimal distance settings
    example_follow_up_2 = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30],
            "edss_score": [4.5, 5.0, 5.5, 4.0],
        }
    )
    example_baselines_2 = example_follow_up_2.iloc[:-1].rename(
        columns={
            "days_after_baseline": "baseline_timestamp",
            "edss_score": "baseline_score",
        }
    )
    # No minimal distance, just a different df
    # NOTE: This looks like PIRA at 20, but actually
    # isn't because this test always uses the previous
    # score as the baseline.
    test_2_target = pd.DataFrame(
        [
            [False, False, False, None, np.nan, 4.5],
            [False, False, False, None, np.nan, 5.0],
            [True, False, True, LABEL_IMPROVEMENT, 4.0, 5.5],
        ]
    )
    test_2_result = pd.DataFrame(
        [
            edssannotation.EDSSAnnotation(
                opt_minimal_distance_time=0,
                opt_require_confirmation=False,
                annotation_mode="experimental-symmetric",
                opt_baseline_type="fixed",
            )._check_assessment_for_progression(
                annotated_df=example_follow_up_2,
                baselines_df=example_baselines_2.iloc[:i],
                current_assessment_index=i,
                additional_lower_threshold=0,
            )
            for i in range(1, 4)
        ]
    )
    assert test_2_result.equals(test_2_target), "Test 2 failed!"
    # Minimal distance to previous
    test_3_target = pd.DataFrame(
        [
            [False, False, False, None, np.nan, 4.5],
            [False, False, False, None, np.nan, 5.0],
            [False, False, False, None, np.nan, 5.5],
        ]
    )
    test_3_result = pd.DataFrame(
        [
            edssannotation.EDSSAnnotation(
                opt_minimal_distance_time=10.1,
                opt_minimal_distance_type="previous",
                opt_require_confirmation=False,
                annotation_mode="experimental-symmetric",
                opt_baseline_type="fixed",
            )._check_assessment_for_progression(
                annotated_df=example_follow_up_2,
                baselines_df=example_baselines_2.iloc[:i],
                current_assessment_index=i,
                additional_lower_threshold=0,
            )
            for i in range(1, 4)
        ]
    )
    assert test_3_result.equals(test_3_target), "Test 3 failed!"
    # Minimal distance to reference, without backtracking
    test_4_target = pd.DataFrame(
        [
            [False, False, False, None, np.nan, 4.5],
            [False, False, False, None, np.nan, 5.0],
            [False, False, False, None, np.nan, 5.5],
        ]
    )
    test_4_result = pd.DataFrame(
        [
            edssannotation.EDSSAnnotation(
                opt_minimal_distance_time=10.1,
                opt_minimal_distance_type="reference",
                opt_minimal_distance_backtrack_decrease=False,
                opt_require_confirmation=False,
                annotation_mode="experimental-symmetric",
                opt_baseline_type="fixed",
            )._check_assessment_for_progression(
                annotated_df=example_follow_up_2,
                baselines_df=example_baselines_2.iloc[:i],
                current_assessment_index=i,
                additional_lower_threshold=0,
            )
            for i in range(1, 4)
        ]
    )
    assert test_4_result.equals(test_4_target), "Test 4 failed!"
    # Minimal distance to reference, with backtracking
    test_5_target = pd.DataFrame(
        [
            [False, False, False, None, np.nan, 4.5],
            [True, True, False, LABEL_PIRA, 5.5, 4.5],
            [True, False, True, LABEL_IMPROVEMENT, 4, 5],
        ]
    )
    test_5_result = pd.DataFrame(
        [
            edssannotation.EDSSAnnotation(
                opt_minimal_distance_time=10.1,
                opt_minimal_distance_type="reference",
                opt_minimal_distance_backtrack_decrease=True,
                opt_require_confirmation=False,
                annotation_mode="experimental-symmetric",
                opt_baseline_type="fixed",
            )._check_assessment_for_progression(
                annotated_df=example_follow_up_2,
                baselines_df=example_baselines_2.iloc[:i],
                current_assessment_index=i,
                additional_lower_threshold=0,
            )
            for i in range(1, 4)
        ]
    )
    assert test_5_result.equals(test_5_target), "Test 5 failed!"


# ----------------------------
# Part 2 - relapse-independent
# -----------------------------


def raw_pira_progression_result_is_equal_to_target(
    follow_up_dataframe,
    targets_dict,
    args_dict={},
):
    annotated_df = edssannotation.EDSSAnnotation(
        **args_dict
    ).add_event_annotation_to_follow_up(
        follow_up_dataframe=follow_up_dataframe,
    )

    # Initialize target dataframe
    target_df = follow_up_dataframe.copy()
    target_df["is_post_event_rebaseline"] = False
    target_df["is_general_rebaseline"] = False
    target_df["edss_score_used_as_new_general_reference"] = np.nan
    target_df["is_event"] = False
    target_df["is_accrual"] = False
    target_df["is_improvement"] = False
    target_df["event_type"] = None
    target_df["event_score"] = np.nan
    target_df["event_reference_score"] = np.nan
    target_df["event_id"] = np.nan
    target_df["accrual_event_id"] = np.nan
    target_df["improvement_event_id"] = np.nan

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
        follow_up_dataframe=test_dataframe_no_next_sustained,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_event": [(30, True)],
            "is_accrual": [(30, True)],
            "event_type": [(30, LABEL_PIRA)],
            "event_score": [(30, 2.0)],
            "event_reference_score": [(30, 1.0)],
            "event_id": [(30, 1.0)],
            "accrual_event_id": [(30, 1.0)],
        },
        args_dict={
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
        },
    ), "Test 1 'unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_no_next_sustained,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_event": [(30, True)],
            "is_accrual": [(30, True)],
            "event_type": [(30, LABEL_PIRA)],
            "event_score": [(30, 2.0)],
            "event_reference_score": [(30, 1.0)],
            "event_id": [(30, 1.0)],
            "accrual_event_id": [(30, 1.0)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 2 'next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
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
        follow_up_dataframe=test_dataframe_durations,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.5)],
            "is_event": [(10, True)],
            "is_accrual": [(10, True)],
            "event_type": [(10, LABEL_PIRA)],
            "event_score": [(10, 2.5)],
            "event_reference_score": [(10, 1.0)],
            "event_id": [(10, 1.0)],
            "accrual_event_id": [(10, 1.0)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 4 '10 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_durations,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_accrual": [(10, True)],
            "event_type": [(10, LABEL_PIRA)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 1.0)],
            "event_id": [(10, 1.0)],
            "accrual_event_id": [(10, 1.0)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 5 '20 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
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
        follow_up_dataframe=test_dataframe_left_tolerance,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_accrual": [(10, True)],
            "event_type": [(10, LABEL_PIRA)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 1.0)],
            "event_id": [(10, 1.0)],
            "accrual_event_id": [(10, 1.0)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 0,
            "opt_baseline_type": "fixed",
        },
    ), "Test 7 '15 units confirmed, no tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_left_tolerance,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.5)],
            "is_event": [(10, True)],
            "is_accrual": [(10, True)],
            "event_type": [(10, LABEL_PIRA)],
            "event_score": [(10, 2.5)],
            "event_reference_score": [(10, 1.0)],
            "event_id": [(10, 1.0)],
            "accrual_event_id": [(10, 1.0)],
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
        follow_up_dataframe=test_dataframe_right_constraint,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_accrual": [(10, True)],
            "event_type": [(10, LABEL_PIRA)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 1.0)],
            "event_id": [(10, 1.0)],
            "accrual_event_id": [(10, 1.0)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_time_right_side_max_tolerance": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 9 '10 units confirmed, 10 units tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_right_constraint,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_event": [(30, True)],
            "is_accrual": [(30, True)],
            "event_type": [(30, LABEL_PIRA)],
            "event_score": [(30, 2.0)],
            "event_reference_score": [(30, 1.0)],
            "event_id": [(30, 1.0)],
            "accrual_event_id": [(30, 1.0)],
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
        follow_up_dataframe=test_dataframe_sustained_minimal_distance,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_accrual": [(10, True)],
            "event_type": [(10, LABEL_PIRA)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 1.0)],
            "event_id": [(10, 1.0)],
            "accrual_event_id": [(10, 1.0)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 11 'Sustained, minimum 20 units' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
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
        follow_up_dataframe=test_dataframe_all_vs_last,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_accrual": [(10, True)],
            "event_type": [(10, LABEL_PIRA)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 1.0)],
            "event_id": [(10, 1.0)],
            "accrual_event_id": [(10, 1.0)],
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
        follow_up_dataframe=test_dataframe_min_vs_monotonic,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_accrual": [(10, True)],
            "event_type": [(10, LABEL_PIRA)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 1.0)],
            "event_id": [(10, 1.0)],
            "accrual_event_id": [(10, 1.0)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_type": "minimum",
            "opt_baseline_type": "fixed",
        },
    ), "Test 15 '30 units confirmed, minimum' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
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
        follow_up_dataframe=test_dataframe_min_vs_monotonic_event_scores,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.5)],
            "is_event": [(10, True)],
            "is_accrual": [(10, True)],
            "event_type": [(10, LABEL_PIRA)],
            "event_score": [(10, 2.5)],
            "event_reference_score": [(10, 1.0)],
            "event_id": [(10, 1.0)],
            "accrual_event_id": [(10, 1.0)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_type": "minimum",
            "opt_baseline_type": "fixed",
        },
    ), "Test 17 '20 units confirmed, minimum' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_min_vs_monotonic_event_scores,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.5)],
            "is_event": [(10, True)],
            "is_accrual": [(10, True)],
            "event_type": [(10, LABEL_PIRA)],
            "event_score": [(10, 2.5)],
            "event_reference_score": [(10, 1.0)],
            "event_id": [(10, 1.0)],
            "accrual_event_id": [(10, 1.0)],
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
        follow_up_dataframe=test_dataframe_last_confirmed,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_event": [(30, True)],
            "is_accrual": [(30, True)],
            "event_type": [(30, LABEL_PIRA)],
            "event_score": [(30, 2.0)],
            "event_reference_score": [(30, 1.0)],
            "event_id": [(30, 1.0)],
            "accrual_event_id": [(30, 1.0)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": True,
        },
    ), "Test 19 'Last requires confirmation' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_last_confirmed,
        targets_dict={
            "is_post_event_rebaseline": [(30, True), (50, True)],
            "is_general_rebaseline": [(30, True), (50, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0), (50, 3.0)],
            "is_event": [(30, True), (50, True)],
            "is_accrual": [(30, True), (50, True)],
            "event_type": [(30, LABEL_PIRA), (50, LABEL_PIRA)],
            "event_score": [(30, 2.0), (50, 3.0)],
            "event_reference_score": [(30, 1.0), (50, 2.0)],
            "event_id": [(30, 1.0), (50, 2.0)],
            "accrual_event_id": [(30, 1.0), (50, 2.0)],
        },
        args_dict={
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": False,
        },
    ), "Test 20 'Last does not require confirmation' failed!"

    # Experimental-inverted mode
    test_dataframe_inv = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40],
            "edss_score": [3.0, 1.0, 1.5, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 1.0)],
            "is_event": [(10, True)],
            "is_improvement": [(10, True)],
            "event_type": [(10, LABEL_IMPROVEMENT)],
            "event_score": [(10, 1.0)],
            "event_reference_score": [(10, 3.0)],
            "event_id": [(10, 1.0)],
            "improvement_event_id": [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": False,
        },
    ), "Test 21 'Inverted unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 1.5)],
            "is_event": [(10, True)],
            "is_improvement": [(10, True)],
            "event_type": [(10, LABEL_IMPROVEMENT)],
            "event_score": [(10, 1.5)],
            "event_reference_score": [(10, 3.0)],
            "event_id": [(10, 1.0)],
            "improvement_event_id": [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 22 'Inverted next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_improvement": [(10, True)],
            "event_type": [(10, LABEL_IMPROVEMENT)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 3.0)],
            "event_id": [(10, 1.0)],
            "improvement_event_id": [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
        },
    ), "Test 23 'Inverted distance-confirmed' failed!"
    test_dataframe_inv_min_mono = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40],
            "edss_score": [3.0, 1.5, 2.0, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_min_mono,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_improvement": [(10, True)],
            "event_type": [(10, LABEL_IMPROVEMENT)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 3.0)],
            "event_id": [(10, 1.0)],
            "improvement_event_id": [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_type": "minimum",
        },
    ), "Test 24 'Inverted next-confirmed, minimum' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_min_mono,
        targets_dict={
            "is_post_event_rebaseline": [(20, True)],
            "is_general_rebaseline": [(20, True)],
            "edss_score_used_as_new_general_reference": [(20, 2.0)],
            "is_event": [(20, True)],
            "is_improvement": [(20, True)],
            "event_type": [(20, LABEL_IMPROVEMENT)],
            "event_score": [(20, 2.0)],
            "event_reference_score": [(20, 3.0)],
            "event_id": [(20, 1.0)],
            "improvement_event_id": [(20, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_type": "monotonic",
        },
    ), "Test 25 'Inverted next-confirmed, monotonic' failed!"
    test_dataframe_inv_all_last = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50],
            "edss_score": [3.0, 1.5, 2.0, 2.0, 1.5, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_all_last,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_improvement": [(10, True)],
            "event_type": [(10, LABEL_IMPROVEMENT)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 3.0)],
            "event_id": [(10, 1.0)],
            "improvement_event_id": [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
        },
    ), "Test 26 'Inverted distance-confirmed, all' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_all_last,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 1.5)],
            "is_event": [(10, True)],
            "is_improvement": [(10, True)],
            "event_type": [(10, LABEL_IMPROVEMENT)],
            "event_score": [(10, 1.5)],
            "event_reference_score": [(10, 3.0)],
            "event_id": [(10, 1.0)],
            "improvement_event_id": [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "last",
        },
    ), "Test 27 'Inverted distance-confirmed, last' failed!"
    test_dataframe_inv_dists = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60],
            "edss_score": [3.0, 3.0, 2.5, 1.0, 1.5, 2.0, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_dists,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 1.0)],
            "is_event": [(30, True)],
            "is_improvement": [(30, True)],
            "event_type": [(30, LABEL_IMPROVEMENT)],
            "event_score": [(30, 1.0)],
            "event_reference_score": [(30, 3.0)],
            "event_id": [(30, 1.0)],
            "improvement_event_id": [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": False,
        },
    ), "Test 28 'Inverted unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_dists,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 1.5)],
            "is_event": [(30, True)],
            "is_improvement": [(30, True)],
            "event_type": [(30, LABEL_IMPROVEMENT)],
            "event_score": [(30, 1.5)],
            "event_reference_score": [(30, 3.0)],
            "event_id": [(30, 1.0)],
            "improvement_event_id": [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 29 'Inverted next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_dists,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_event": [(30, True)],
            "is_improvement": [(30, True)],
            "event_type": [(30, LABEL_IMPROVEMENT)],
            "event_score": [(30, 2.0)],
            "event_reference_score": [(30, 3.0)],
            "event_id": [(30, 1.0)],
            "improvement_event_id": [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
        },
    ), "Test 30 'Inverted distance-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_dists,
        targets_dict={},
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
        },
    ), "Test 31 'Inverted sustained' failed!"
    test_dataframe_inv_left = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40],
            "edss_score": [3.0, 1.5, 1.5, 2.0, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_left,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_improvement": [(10, True)],
            "event_type": [(10, LABEL_IMPROVEMENT)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 3.0)],
            "event_id": [(10, 1.0)],
            "improvement_event_id": [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 0,
        },
    ), "Test 32 'Inverted no left-hand tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_left,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 1.5)],
            "is_event": [(10, True)],
            "is_improvement": [(10, True)],
            "event_type": [(10, LABEL_IMPROVEMENT)],
            "event_score": [(10, 1.5)],
            "event_reference_score": [(10, 3.0)],
            "event_id": [(10, 1.0)],
            "improvement_event_id": [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 5,
        },
    ), "Test 33 'Inverted with left-hand tolerance' failed!"
    test_dataframe_inv_right = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 30, 40],
            "edss_score": [3.0, 1.5, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_right,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_improvement": [(10, True)],
            "event_type": [(10, LABEL_IMPROVEMENT)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 3.0)],
            "event_id": [(10, 1.0)],
            "improvement_event_id": [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_time_right_side_max_tolerance": np.inf,
        },
    ), "Test 34 'Inverted no right-hand constraint' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_right,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_event": [(30, True)],
            "is_improvement": [(30, True)],
            "event_type": [(30, LABEL_IMPROVEMENT)],
            "event_score": [(30, 2.0)],
            "event_reference_score": [(30, 3.0)],
            "event_id": [(30, 1.0)],
            "improvement_event_id": [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_time_right_side_max_tolerance": 5,
        },
    ), "Test 35 'Inverted with right-hand constraint' failed!"
    test_dataframe_inv_left_right = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 40],
            "edss_score": [3.0, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_left_right,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_improvement": [(10, True)],
            "event_type": [(10, LABEL_IMPROVEMENT)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 3.0)],
            "event_id": [(10, 1.0)],
            "improvement_event_id": [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_time_left_side_max_tolerance": 5,
            "opt_confirmation_time_right_side_max_tolerance": 10,
        },
    ), "Test 36 'Inverted left-hand tolerance and right-hand constraint' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_left_right,
        targets_dict={},
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 0,
            "opt_confirmation_time_right_side_max_tolerance": 10,
        },
    ), "Test 37 'Inverted no left-hand tolerance but right-hand constraint' failed!"
    test_dataframe_inv_sust = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30],
            "edss_score": [3.0, 1.5, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_sust,
        targets_dict={
            "is_post_event_rebaseline": [(10, True)],
            "is_general_rebaseline": [(10, True)],
            "edss_score_used_as_new_general_reference": [(10, 2.0)],
            "is_event": [(10, True)],
            "is_improvement": [(10, True)],
            "event_type": [(10, LABEL_IMPROVEMENT)],
            "event_score": [(10, 2.0)],
            "event_reference_score": [(10, 3.0)],
            "event_id": [(10, 1.0)],
            "improvement_event_id": [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 20,
        },
    ), "Test 38 'Inverted sustained minimal distance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_sust,
        targets_dict={},
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 21,
        },
    ), "Test 39 'Inverted sustained minimal distance' failed!"
    test_dataframe_last_ext = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50],
            "edss_score": [3.0, 3.0, 2.5, 2.0, 1.5, 1.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_last_ext,
        targets_dict={
            "is_post_event_rebaseline": [(30, True)],
            "is_general_rebaseline": [(30, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0)],
            "is_event": [(30, True)],
            "is_improvement": [(30, True)],
            "event_type": [(30, LABEL_IMPROVEMENT)],
            "event_score": [(30, 2.0)],
            "event_reference_score": [(30, 3.0)],
            "event_id": [(30, 1.0)],
            "improvement_event_id": [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": True,
        },
    ), "Test 40 'Inverted, last must be confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_last_ext,
        targets_dict={
            "is_post_event_rebaseline": [(30, True), (50, True)],
            "is_general_rebaseline": [(30, True), (50, True)],
            "edss_score_used_as_new_general_reference": [(30, 2.0), (50, 1.0)],
            "is_event": [(30, True), (50, True)],
            "is_improvement": [(30, True), (50, True)],
            "event_type": [(30, LABEL_IMPROVEMENT), (50, LABEL_IMPROVEMENT)],
            "event_score": [(30, 2.0), (50, 1.0)],
            "event_reference_score": [(30, 3.0), (50, 2.0)],
            "event_id": [(30, 1.0), (50, 2.0)],
            "improvement_event_id": [(30, 1.0), (50, 2.0)],
        },
        args_dict={
            "annotation_mode": "experimental-inverted",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": False,
        },
    ), "Test 41 'Inverted, last must not be confirmed' failed!"


if __name__ == "__main__":
    print("\nPart 1 - building blocks\n")
    print("Testing '_is_large_enough_increase_or_decrease'...")
    test_is_large_enough_increase_or_decrease()

    print("Testing '_get_confirmation_scores_dataframe'...")
    test_get_confirmation_scores_dataframe()

    print("Testing '_check_confirmation_scores_and_get_confirmed_score'...")
    test_check_confirmation_scores_and_get_confirmed_score()

    print("Testing '_backtrack_minimal_distance_compatible_reference'...")
    test_backtrack_minimal_distance_compatible_reference()

    print("Testing '_check_assessment_for_progression'...")
    test_check_assessment_for_progression()

    print("\nPart 2 - relapse independent progression\n")
    print("Testing confirmation...")
    test_relapse_independent_confirmation()

    """
    print("Testing baselines...")
    test_relapse_independent_baselines()

    print("Testing minimal distance...")
    test_relapse_independent_minimal_distance()

    print("Testing first vs. all events...")
    test_relapse_independent_first_vs_all_events()

    print("Testing multiple events re-baselining...")
    test_relapse_independent_multiple_events_rebaselining()
    """

    # print("\nPart 3 - progression with relapses\n")

    # print("\nPart 4 - multi-event mode\n")
    # print("Testing multi-event mode...")
    # test_multi_event_option()

    print("\nAll tests successfully completed.\n")
