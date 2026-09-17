"""Test edssannotation.py functionality.

Execute

    python -m tests.tests

in parent to run tests!

This is a bit messy for now. Use tools.visualization.py for some
visual testing.

"""

from collections import Counter

import numpy as np
import pandas as pd

from definitions import edssannotation

# Set some globals
LABEL_PIRA = "PIRA"
LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW = "PIRA with relapse during confirmation"
LABEL_RAW = "RAW"
LABEL_UNDEFINED = "Undefined"
LABEL_IMPROVEMENT = "Improvement"

ACCRUAL_MODE_NAME = "accrual"
IMPROVEMENT_MODE_NAME = "improvement"
SYMMETRIC_MODE_NAME = "symmetric"

BASELINE_TIMESTAMP = "baseline_timestamp"
BASELINE_SCORE = "baseline_score"

TIMESTAMP = "days_after_baseline"
EDSS_SCORE = "edss_score"
DAYS_TO_NEXT_RELAPSE = "days_to_next_relapse"
DAYS_SINCE_PREVIOUS_RELAPSE = "days_since_previous_relapse"
IS_POST_EVENT_REBASELINE = "is_post_event_rebaseline"
IS_POST_RELAPSE_REBASELINE = "is_post_relapse_rebaseline"
IS_GENERAL_REBASELINE = "is_general_rebaseline"
IS_PIRA_REBASELINE = "is_pira_rebaseline"
EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE = "edss_score_used_as_new_general_reference"
EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE = "edss_score_used_as_new_pira_reference"
IS_EVENT = "is_event"
IS_ACCRUAL_EVENT = "is_accrual_event"
IS_IMPROVEMENT_EVENT = "is_improvement_event"
EVENT_TYPE = "event_type"
EVENT_SCORE = "event_score"
EVENT_REFERENCE_SCORE = "event_reference_score"
EVENT_ID = "event_id"
ACCRUAL_EVENT_ID = "accrual_event_id"
IMPROVEMENT_EVENT_ID = "improvement_event_id"


# Test the increase/decrease delta check
def test_is_large_enough_increase_or_decrease():
    current_name = "current"
    reference_name = "reference"
    target_name = "target"

    # Test 1 - minimum required increase + 0.5 irrespective of reference
    test_cases_1 = [
        {current_name: 0, reference_name: 0, target_name: [False, False]},
        # Increase
        {current_name: 0.5, reference_name: 0, target_name: [True, False]},
        {current_name: 1.0, reference_name: 0, target_name: [True, False]},
        {current_name: 1.5, reference_name: 0, target_name: [True, False]},
        {current_name: 2.0, reference_name: 2.0, target_name: [False, False]},
        {current_name: 2.5, reference_name: 2.0, target_name: [True, False]},
        {current_name: 3.0, reference_name: 2.0, target_name: [True, False]},
        # Decrease
        {current_name: 0, reference_name: 0.5, target_name: [False, True]},
        {current_name: 0, reference_name: 1.0, target_name: [False, True]},
        {current_name: 0, reference_name: 1.5, target_name: [False, True]},
        {current_name: 2.0, reference_name: 2.0, target_name: [False, False]},
        {current_name: 9.5, reference_name: 10.0, target_name: [False, True]},
        {current_name: 5.5, reference_name: 6.0, target_name: [False, True]},
        {current_name: 5.0, reference_name: 6.0, target_name: [False, True]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    opt_max_score_that_requires_plus_1=-1,
                    opt_larger_increment_from_0=False,
                )._is_large_enough_increase_or_decrease(
                    current_edss=test_case[current_name],
                    reference_edss=test_case[reference_name],
                )
                for test_case in test_cases_1
            ]
        ),
        np.array([test_case[target_name] for test_case in test_cases_1]),
        err_msg="Minimal increase + 0.5 irrespective of baseline failed!",
    )
    # Test 2 - minimum required increase + 1.0 irrespective of reference
    test_cases_2 = [
        {current_name: 0, reference_name: 0, target_name: [False, False]},
        # Increase
        {current_name: 0.5, reference_name: 0, target_name: [False, False]},
        {current_name: 1.0, reference_name: 0, target_name: [True, False]},
        {current_name: 1.5, reference_name: 0, target_name: [True, False]},
        {current_name: 2.0, reference_name: 2.0, target_name: [False, False]},
        {current_name: 2.5, reference_name: 2.0, target_name: [False, False]},
        {current_name: 3.0, reference_name: 2.0, target_name: [True, False]},
        {current_name: 10.0, reference_name: 9.5, target_name: [False, False]},
        # Decrease
        {current_name: 0, reference_name: 0.5, target_name: [False, False]},
        {current_name: 0, reference_name: 1.0, target_name: [False, True]},
        {current_name: 0, reference_name: 1.5, target_name: [False, True]},
        {current_name: 9.5, reference_name: 10.0, target_name: [False, False]},
        {current_name: 5.5, reference_name: 6.0, target_name: [False, False]},
        {current_name: 5.0, reference_name: 6.0, target_name: [False, True]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    opt_max_score_that_requires_plus_1=10.0,
                    opt_larger_increment_from_0=False,
                )._is_large_enough_increase_or_decrease(
                    current_edss=test_case[current_name],
                    reference_edss=test_case[reference_name],
                )
                for test_case in test_cases_2
            ]
        ),
        np.array([test_case[target_name] for test_case in test_cases_2]),
        err_msg="Minimal increase + 1.0 irrespective of baseline failed!",
    )
    # Test 3 - minimum required increase from 0 + 1.5
    test_cases_3 = [
        {current_name: 0, reference_name: 0, target_name: [False, False]},
        # Increase
        {current_name: 0.5, reference_name: 0, target_name: [False, False]},
        {current_name: 1.0, reference_name: 0, target_name: [False, False]},
        {current_name: 1.5, reference_name: 0, target_name: [True, False]},
        {current_name: 2.0, reference_name: 2.0, target_name: [False, False]},
        {current_name: 2.5, reference_name: 2.0, target_name: [True, False]},
        {current_name: 3.0, reference_name: 2.0, target_name: [True, False]},
        # Decrease
        {current_name: 0, reference_name: 0.5, target_name: [False, False]},
        {current_name: 0, reference_name: 1.0, target_name: [False, False]},
        {current_name: 0, reference_name: 1.5, target_name: [False, True]},
        {current_name: 2.0, reference_name: 2.0, target_name: [False, False]},
        {current_name: 2.0, reference_name: 2.5, target_name: [False, True]},
        {current_name: 2.0, reference_name: 3.0, target_name: [False, True]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    opt_max_score_that_requires_plus_1=1,
                    opt_larger_increment_from_0=True,
                )._is_large_enough_increase_or_decrease(
                    current_edss=test_case[current_name],
                    reference_edss=test_case[reference_name],
                )
                for test_case in test_cases_3
            ]
        ),
        np.array([test_case[target_name] for test_case in test_cases_3]),
        err_msg="Minimal increase + 1.5 from 0 failed!",
    )
    # Test 4 - standard case
    test_cases_4 = [
        {current_name: 0, reference_name: 0, target_name: [False, False]},
        # Increase
        {current_name: 0.5, reference_name: 0, target_name: [False, False]},
        {current_name: 1.0, reference_name: 0, target_name: [False, False]},
        {current_name: 1.5, reference_name: 0, target_name: [True, False]},
        {current_name: 2.0, reference_name: 2.0, target_name: [False, False]},
        {current_name: 2.5, reference_name: 2.0, target_name: [False, False]},
        {current_name: 3.0, reference_name: 2.0, target_name: [True, False]},
        {current_name: 5.5, reference_name: 5.0, target_name: [False, False]},
        {current_name: 6.0, reference_name: 5.0, target_name: [True, False]},
        {current_name: 6.0, reference_name: 5.5, target_name: [True, False]},
        {current_name: 6.5, reference_name: 5.5, target_name: [True, False]},
        # Decrease
        {current_name: 0, reference_name: 0.5, target_name: [False, False]},
        {current_name: 0, reference_name: 1.0, target_name: [False, False]},
        {current_name: 0, reference_name: 1.5, target_name: [False, True]},
        {current_name: 9.5, reference_name: 10.0, target_name: [False, True]},
        {current_name: 5.5, reference_name: 6.0, target_name: [False, True]},
        {current_name: 5.0, reference_name: 6.0, target_name: [False, True]},
        {current_name: 5.0, reference_name: 5.5, target_name: [False, False]},
        {current_name: 4.5, reference_name: 5.5, target_name: [False, True]},
        {current_name: 4.5, reference_name: 5.0, target_name: [False, False]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation()._is_large_enough_increase_or_decrease(
                    current_edss=test_case[current_name],
                    reference_edss=test_case[reference_name],
                )
                for test_case in test_cases_4
            ]
        ),
        np.array([test_case[target_name] for test_case in test_cases_4]),
        err_msg="Standard minimal increase settings failed!",
    )


# Test the confirmation dataframe extraction
def test_get_confirmation_scores_dataframe():
    test_dataframe = pd.DataFrame(
        {TIMESTAMP: [0, 10, 20, 30, 40, 50], "score": [0, 1, 2, 3, 4, 5]}
    )
    # Test case 1 - sustained, assessments available
    # Must yield all assessments following the first one.
    test_case_1 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
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
    test_case_2 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
        current_timestamp=50,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=-1,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=0,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_2.equals(test_dataframe[test_dataframe[TIMESTAMP] > 50]), (
        "Test 2 failed!"
    )
    # Test case 3 - sustained, minimal time interval
    # Must yield all assessments following the first one.
    test_case_3 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
        current_timestamp=0,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=-1,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=15,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_3.equals(test_dataframe.iloc[1:]), "Test 3 failed!"
    # Test case 3b - sustained, minimal time interval, too long.
    # Must yield an empty dataframe.
    test_case_3 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
        current_timestamp=0,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=-1,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=51,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_3.equals(test_dataframe[test_dataframe[TIMESTAMP] > 50]), (
        "Test 3b failed!"
    )
    # Test case 4 - sustained, minimal time interval, no assessments available
    # Must yield an empty dataframe.
    test_case_4 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
        current_timestamp=40,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=-1,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=15,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_4.equals(test_dataframe[test_dataframe[TIMESTAMP] > 50]), (
        "Test 4 failed!"
    )
    # Test case 5 - time interval, assessments available
    # Must yield all assessments following the first one
    # until and including the assessment at 40.
    test_case_5 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
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
    test_case_6 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
        current_timestamp=10,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=50,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=0,
        opt_confirmation_time_right_side_max_tolerance=np.inf,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_6.equals(test_dataframe[test_dataframe[TIMESTAMP] > 50]), (
        "Test 6 failed!"
    )
    # Test case 7 - time interval, right side constrained
    # Next is further away, but within tolerance.
    test_case_7 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
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
    test_case_8 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
        current_timestamp=10,
        follow_up_dataframe=test_dataframe,
        opt_confirmation_time=5,
        opt_confirmation_included_values="all",
        opt_confirmation_sustained_minimal_distance=10,
        opt_confirmation_time_right_side_max_tolerance=4,
        opt_confirmation_time_left_side_max_tolerance=0,
    )
    assert test_case_8.equals(test_dataframe[test_dataframe[TIMESTAMP] > 50]), (
        "Test 8 failed!"
    )
    # Test case 9 - time interval, left side tolerance standard
    # Must yield assesments at 20 and 30 (30 is first).
    test_case_9 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
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
    test_case_10 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
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
    test_case_11 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
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
    test_case_12 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
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
    test_case_13 = edssannotation.EDSSAnnotation()._get_confirmation_scores_dataframe(
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
    test_dataframe = pd.DataFrame({EDSS_SCORE: [5, 4.5, 5]})
    # Test case 1 - minimum, against a reference of 4
    # Not confirmed, confirmed score is nan.
    assert edssannotation.EDSSAnnotation(
        opt_confirmation_type="minimum",
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=5,
        current_reference=4,
        confirmation_scores_dataframe=test_dataframe,
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
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=5,
        current_reference=3.5,
        confirmation_scores_dataframe=test_dataframe,
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
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=5,
        current_reference=3.5,
        confirmation_scores_dataframe=test_dataframe,
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
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=5,
        current_reference=4.0,
        confirmation_scores_dataframe=pd.DataFrame({EDSS_SCORE: [5]}),
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
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=5,
        current_reference=4.0,
        confirmation_scores_dataframe=pd.DataFrame({EDSS_SCORE: []}),
    ) == (
        False,
        False,
        np.nan,
    ), "Test 5 failed!"

    # Part 2 - decrease
    test_dataframe = pd.DataFrame({EDSS_SCORE: [4, 4.5, 4]})
    # Test case 1 - minimum, against a reference of 5
    # Not confirmed, confirmed score is nan.
    assert edssannotation.EDSSAnnotation(
        opt_confirmation_type="minimum",
        opt_max_score_that_requires_plus_1=5.0,
        opt_larger_increment_from_0=True,
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=4,
        current_reference=5,
        confirmation_scores_dataframe=test_dataframe,
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
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=4,
        current_reference=5.5,
        confirmation_scores_dataframe=test_dataframe,
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
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=4,
        current_reference=5.5,
        confirmation_scores_dataframe=test_dataframe,
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
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=4,
        current_reference=5,
        confirmation_scores_dataframe=pd.DataFrame({EDSS_SCORE: [4]}),
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
    )._check_confirmation_scores_and_get_confirmed_score(
        current_edss=5,
        current_reference=4.0,
        confirmation_scores_dataframe=pd.DataFrame({EDSS_SCORE: []}),
    ) == (
        False,
        False,
        np.nan,
    ), "Test 10 failed!"


def test_backtrack_minimal_distance_compatible_reference():
    test_dataframe = pd.DataFrame(
        {
            BASELINE_TIMESTAMP: [0, 10, 20, 30, 40, 50],
            BASELINE_SCORE: [6.0, 5.5, 5.5, 5.0, 5.0, 4.5],
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
                        BASELINE_TIMESTAMP: [0, 10, 20],
                        BASELINE_SCORE: [3.5, 3.0, 2.5],
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
                        BASELINE_TIMESTAMP: [0, 10, 20],
                        BASELINE_SCORE: [3.5, 3.0, 2.5],
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
                        BASELINE_TIMESTAMP: [0, 10, 20],
                        BASELINE_SCORE: [4.5, 5.0, 5.5],
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
                        BASELINE_TIMESTAMP: [0, 10, 20],
                        BASELINE_SCORE: [4.5, 5.0, 5.5],
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


def test_check_assessment_for_event():
    # Returns is_event, is_accrual_event, is_improvement_event,
    # event_type, confirmed_event_score,
    # current_baseline_score

    # Without minimal distance, without confirmation
    example_follow_up_1 = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [4.5, 5.5, 5.5, 4.0],
            DAYS_TO_NEXT_RELAPSE: [np.nan for _ in range(4)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [np.nan for _ in range(4)],
        }
    )
    example_baselines_1 = example_follow_up_1.iloc[:-1].rename(
        columns={
            TIMESTAMP: BASELINE_TIMESTAMP,
            EDSS_SCORE: BASELINE_SCORE,
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
                annotation_mode=SYMMETRIC_MODE_NAME,
                opt_baseline_type="fixed",
            )._check_assessment_for_event(
                annotated_df=example_follow_up_1,
                relapse_timestamps=[],
                baselines_df=example_baselines_1.iloc[:i],
                current_assessment_index=i,
                check_pira=[True, True, False, False][i],
                check_raw=[False, False, True, True][i],
                check_undefined=[False, False, True, True][i],
            )
            for i in range(1, 4)
        ]
    )
    assert test_1_result.equals(test_1_target), "Test 1 failed!"

    # With different minimal distance settings
    example_follow_up_2 = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [4.5, 5.0, 5.5, 4.0],
            DAYS_TO_NEXT_RELAPSE: [np.nan for _ in range(4)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [np.nan for _ in range(4)],
        }
    )
    example_baselines_2 = example_follow_up_2.iloc[:-1].rename(
        columns={
            TIMESTAMP: BASELINE_TIMESTAMP,
            EDSS_SCORE: BASELINE_SCORE,
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
                annotation_mode=SYMMETRIC_MODE_NAME,
                opt_baseline_type="fixed",
            )._check_assessment_for_event(
                annotated_df=example_follow_up_2,
                relapse_timestamps=[],
                baselines_df=example_baselines_2.iloc[:i],
                current_assessment_index=i,
                check_pira=[True, True, True, False][i],
                check_raw=[False, False, False, True][i],
                check_undefined=[False, False, False, True][i],
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
                annotation_mode=SYMMETRIC_MODE_NAME,
                opt_baseline_type="fixed",
            )._check_assessment_for_event(
                annotated_df=example_follow_up_2,
                relapse_timestamps=[],
                baselines_df=example_baselines_2.iloc[:i],
                current_assessment_index=i,
                check_pira=True,
                check_raw=False,
                check_undefined=False,
            )
            for i in range(1, 4)
        ]
    )
    assert test_3_result.equals(test_3_target), "Test 3 failed!"
    test_3b_result = pd.DataFrame(
        [
            edssannotation.EDSSAnnotation(
                opt_minimal_distance_time=10.1,
                opt_minimal_distance_type="previous",
                opt_require_confirmation=False,
                annotation_mode=SYMMETRIC_MODE_NAME,
                opt_baseline_type="fixed",
            )._check_assessment_for_event(
                annotated_df=example_follow_up_2,
                relapse_timestamps=[],
                baselines_df=example_baselines_2.iloc[:i],
                current_assessment_index=i,
                check_pira=False,
                check_raw=True,
                check_undefined=True,
            )
            for i in range(1, 4)
        ]
    )
    assert test_3b_result.equals(test_3_target), "Test 3b failed!"
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
                annotation_mode=SYMMETRIC_MODE_NAME,
                opt_baseline_type="fixed",
            )._check_assessment_for_event(
                annotated_df=example_follow_up_2,
                relapse_timestamps=[],
                baselines_df=example_baselines_2.iloc[:i],
                current_assessment_index=i,
                check_pira=True,
                check_raw=False,
                check_undefined=False,
            )
            for i in range(1, 4)
        ]
    )
    assert test_4_result.equals(test_4_target), "Test 4 failed!"
    test_4b_result = pd.DataFrame(
        [
            edssannotation.EDSSAnnotation(
                opt_minimal_distance_time=10.1,
                opt_minimal_distance_type="reference",
                opt_minimal_distance_backtrack_decrease=False,
                opt_require_confirmation=False,
                annotation_mode=SYMMETRIC_MODE_NAME,
                opt_baseline_type="fixed",
            )._check_assessment_for_event(
                annotated_df=example_follow_up_2,
                relapse_timestamps=[],
                baselines_df=example_baselines_2.iloc[:i],
                current_assessment_index=i,
                check_pira=False,
                check_raw=True,
                check_undefined=True,
            )
            for i in range(1, 4)
        ]
    )
    assert test_4b_result.equals(test_4_target), "Test 4b failed!"
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
                annotation_mode=SYMMETRIC_MODE_NAME,
                opt_baseline_type="fixed",
            )._check_assessment_for_event(
                annotated_df=example_follow_up_2,
                relapse_timestamps=[],
                baselines_df=example_baselines_2.iloc[:i],
                current_assessment_index=i,
                check_pira=[True, True, True, False][i],
                check_raw=[False, False, False, True][i],
                check_undefined=[False, False, False, True][i],
            )
            for i in range(1, 4)
        ]
    )
    assert test_5_result.equals(test_5_target), "Test 5 failed!"


# ----------------------------
# Part 2 - relapse-independent
# ----------------------------


def raw_pira_progression_result_is_equal_to_target(
    follow_up_dataframe,
    targets_dict,
    relapse_timestamps=None,
    args_dict=None,
):
    if relapse_timestamps is None:
        relapse_timestamps = []
    if args_dict is None:
        args_dict = {}
    annotated_df = edssannotation.EDSSAnnotation(
        **args_dict
    ).add_event_annotation_to_follow_up(
        follow_up_dataframe=follow_up_dataframe,
        relapse_timestamps=relapse_timestamps,
    )

    # Initialize target dataframe
    # NOTE: DO NOT CHANGE COLUMN ORDER!
    target_df = follow_up_dataframe.copy()
    target_df[DAYS_SINCE_PREVIOUS_RELAPSE] = np.nan
    target_df[DAYS_TO_NEXT_RELAPSE] = np.nan
    target_df[IS_POST_EVENT_REBASELINE] = False
    target_df[IS_GENERAL_REBASELINE] = False
    if args_dict.get("annotation_mode", ACCRUAL_MODE_NAME) in [
        ACCRUAL_MODE_NAME,
        SYMMETRIC_MODE_NAME,
    ]:
        target_df[IS_PIRA_REBASELINE] = False
    target_df[EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE] = np.nan
    if args_dict.get("annotation_mode", ACCRUAL_MODE_NAME) in [
        ACCRUAL_MODE_NAME,
        SYMMETRIC_MODE_NAME,
    ]:
        target_df[EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE] = np.nan
    target_df[IS_EVENT] = False
    target_df[IS_ACCRUAL_EVENT] = False
    target_df[IS_IMPROVEMENT_EVENT] = False
    target_df[EVENT_TYPE] = None
    target_df[EVENT_SCORE] = np.nan
    target_df[EVENT_REFERENCE_SCORE] = np.nan
    target_df[EVENT_ID] = np.nan
    target_df[ACCRUAL_EVENT_ID] = np.nan
    target_df[IMPROVEMENT_EVENT_ID] = np.nan
    if args_dict.get("annotation_mode", ACCRUAL_MODE_NAME) in [
        ACCRUAL_MODE_NAME,
        SYMMETRIC_MODE_NAME,
    ]:
        target_df[IS_POST_RELAPSE_REBASELINE] = False

    target_df = target_df.set_index(TIMESTAMP)
    for target_column in targets_dict:
        for target in targets_dict.get(target_column, []):
            target_df.at[target[0], target_column] = target[1]
    target_df = target_df.reset_index()

    return annotated_df.equals(target_df)


def test_relapse_independent_confirmation():
    # Unconfirmed vs. next-confirmed vs. sustained
    test_dataframe_no_next_sustained = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50],
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_no_next_sustained,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
            "opt_baseline_type": "fixed",
        },
    ), "Test 1 'unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_no_next_sustained,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 2 'next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_no_next_sustained,
        targets_dict={},
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
        },
    ), "Test 3 'sustained' failed!"

    # Test various confirmation durations
    test_dataframe_durations = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [1, 2.5, 2.5, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_durations,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.5)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.5)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.5)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 4 '10 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_durations,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 5 '20 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_durations,
        targets_dict={},
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_baseline_type": "fixed",
        },
    ), "Test 6 '30 units confirmed' failed!"

    # Test left-hand side tolerance
    test_dataframe_left_tolerance = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [1, 2.5, 2.5, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_left_tolerance,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 0,
            "opt_baseline_type": "fixed",
        },
    ), "Test 7 '15 units confirmed, no tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_left_tolerance,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.5)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.5)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.5)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 8 '15 units confirmed, 5 units tolerance' failed!"

    # Test right-hand side max. distance constraint
    test_dataframe_right_constraint = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 30, 40],
            EDSS_SCORE: [1, 2.5, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_right_constraint,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_time_right_side_max_tolerance": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 9 '10 units confirmed, 10 units tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_right_constraint,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_time_right_side_max_tolerance": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 10 '10 units confirmed, 5 units tolerance' failed!"

    # Test minimal distance for sustained
    test_dataframe_sustained_minimal_distance = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [1, 2.5, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_sustained_minimal_distance,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
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
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 21,
            "opt_baseline_type": "fixed",
        },
    ), "Test 12 'Sustained, minimum 21 units' failed!"
    # Minimum distance must still include all scores before
    test_dataframe_sustained_minimal_distance_2 = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [3.0, 1.5, 3.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_sustained_minimal_distance_2,
        targets_dict={},
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 0,
            "opt_baseline_type": "fixed",
        },
    ), "Test 12b 'Sustained, no minimum distance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_sustained_minimal_distance_2,
        targets_dict={},
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 15,
            "opt_baseline_type": "fixed",
        },
    ), "Test 12c 'Sustained, no minimum distance' failed!"

    # Test all vs. last confirmed
    test_dataframe_all_vs_last = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [1, 2.0, 1.5, 1.5, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_all_vs_last,
        targets_dict={},
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "opt_baseline_type": "fixed",
        },
    ), "Test 13 '30 units confirmed, all values' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_all_vs_last,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "last",
            "opt_baseline_type": "fixed",
        },
    ), "Test 14 '30 units confirmed, last only' failed!"

    # Minimum vs. monotonic
    test_dataframe_min_vs_monotonic = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [1, 2.5, 2.0, 2.0, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_min_vs_monotonic,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
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
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_type": "monotonic",
            "opt_baseline_type": "fixed",
        },
    ), "Test 16 '30 units confirmed, monotonic' failed!"

    # Minimum/monotonic - correct event scores?
    test_dataframe_min_vs_monotonic_event_scores = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [1, 2.5, 2.5, 3.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_min_vs_monotonic_event_scores,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.5)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.5)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.5)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_type": "minimum",
            "opt_baseline_type": "fixed",
        },
    ), "Test 17 '20 units confirmed, minimum' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_min_vs_monotonic_event_scores,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.5)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.5)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.5)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_type": "monotonic",
            "opt_baseline_type": "fixed",
        },
    ), "Test 18 '20 units confirmed, monotonic' failed!"

    # No confirmation requirement for last assessment
    test_dataframe_last_confirmed = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50],
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.5, 3.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_last_confirmed,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": True,
            "opt_baseline_type": "fixed",
        },
    ), "Test 19 'Last requires confirmation' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_last_confirmed,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (50, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True)],
            IS_PIRA_REBASELINE: [(30, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0), (50, 3.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (50, 3.0)],
            IS_EVENT: [(30, True), (50, True)],
            IS_ACCRUAL_EVENT: [(30, True), (50, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (50, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0), (50, 3.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (50, 2.0)],
            EVENT_ID: [(30, 1.0), (50, 2.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0), (50, 2.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 20 'Last does not require confirmation' failed!"

    # Experimental-improvement mode
    test_dataframe_inv = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [3.0, 1.0, 1.5, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            # IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 1.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 1.0)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 1.0)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 21 'Inverted unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            # IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 1.5)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 1.5)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 1.5)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 22 'Inverted next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            # IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 23 'Inverted distance-confirmed' failed!"
    test_dataframe_inv_min_mono = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [3.0, 1.5, 2.0, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_min_mono,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            # IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_type": "minimum",
            "opt_baseline_type": "fixed",
        },
    ), "Test 24 'Inverted next-confirmed, minimum' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_min_mono,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(20, True)],
            IS_GENERAL_REBASELINE: [(20, True)],
            # IS_PIRA_REBASELINE: [(20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
            IS_EVENT: [(20, True)],
            IS_IMPROVEMENT_EVENT: [(20, True)],
            EVENT_TYPE: [(20, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(20, 2.0)],
            EVENT_REFERENCE_SCORE: [(20, 3.0)],
            EVENT_ID: [(20, 1.0)],
            IMPROVEMENT_EVENT_ID: [(20, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_type": "monotonic",
            "opt_baseline_type": "fixed",
        },
    ), "Test 25 'Inverted next-confirmed, monotonic' failed!"
    test_dataframe_inv_all_last = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50],
            EDSS_SCORE: [3.0, 1.5, 2.0, 2.0, 1.5, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_all_last,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            # IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "opt_baseline_type": "fixed",
        },
    ), "Test 26 'Inverted distance-confirmed, all' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_all_last,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            # IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 1.5)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 1.5)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 1.5)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "last",
            "opt_baseline_type": "fixed",
        },
    ), "Test 27 'Inverted distance-confirmed, last' failed!"
    test_dataframe_inv_dists = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
            EDSS_SCORE: [3.0, 3.0, 2.5, 1.0, 1.5, 2.0, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_dists,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            # IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 1.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 1.0)],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 1.0)],
            EVENT_REFERENCE_SCORE: [(30, 3.0)],
            EVENT_ID: [(30, 1.0)],
            IMPROVEMENT_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 28 'Inverted unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_dists,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            # IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 1.5)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 1.5)],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 1.5)],
            EVENT_REFERENCE_SCORE: [(30, 3.0)],
            EVENT_ID: [(30, 1.0)],
            IMPROVEMENT_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 29 'Inverted next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_dists,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            # IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 3.0)],
            EVENT_ID: [(30, 1.0)],
            IMPROVEMENT_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 30 'Inverted distance-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_dists,
        targets_dict={},
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_baseline_type": "fixed",
        },
    ), "Test 31 'Inverted sustained' failed!"
    test_dataframe_inv_left = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [3.0, 1.5, 1.5, 2.0, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_left,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            # IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 0,
            "opt_baseline_type": "fixed",
        },
    ), "Test 32 'Inverted no left-hand tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_left,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            # IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 1.5)],
            # "edss_score_used_as_new_raw_pira_reference": [(10, 1.5)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 1.5)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 33 'Inverted with left-hand tolerance' failed!"
    test_dataframe_inv_right = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 30, 40],
            EDSS_SCORE: [3.0, 1.5, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_right,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            # IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_time_right_side_max_tolerance": np.inf,
            "opt_baseline_type": "fixed",
        },
    ), "Test 34 'Inverted no right-hand constraint' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_right,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            # IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 3.0)],
            EVENT_ID: [(30, 1.0)],
            IMPROVEMENT_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_time_right_side_max_tolerance": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 35 'Inverted with right-hand constraint' failed!"
    test_dataframe_inv_left_right = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 40],
            EDSS_SCORE: [3.0, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_left_right,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            # IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_time_left_side_max_tolerance": 5,
            "opt_confirmation_time_right_side_max_tolerance": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 36 'Inverted left-hand tolerance and right-hand constraint' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_left_right,
        targets_dict={},
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 0,
            "opt_confirmation_time_right_side_max_tolerance": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 37 'Inverted no left-hand tolerance but right-hand constraint' failed!"
    test_dataframe_inv_sust = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [3.0, 1.5, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_sust,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            # IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 38 'Inverted sustained minimal distance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_sust,
        targets_dict={},
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 21,
            "opt_baseline_type": "fixed",
        },
    ), "Test 39 'Inverted sustained minimal distance' failed!"
    test_dataframe_last_ext = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50],
            EDSS_SCORE: [3.0, 3.0, 2.5, 2.0, 1.5, 1.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_last_ext,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            # IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 3.0)],
            EVENT_ID: [(30, 1.0)],
            IMPROVEMENT_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": True,
            "opt_baseline_type": "fixed",
        },
    ), "Test 40 'Inverted, last must be confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_last_ext,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (50, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True)],
            # IS_PIRA_REBASELINE: [(30, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0), (50, 1.0)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (50, 1.0)],
            IS_EVENT: [(30, True), (50, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (50, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT), (50, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 2.0), (50, 1.0)],
            EVENT_REFERENCE_SCORE: [(30, 3.0), (50, 2.0)],
            EVENT_ID: [(30, 1.0), (50, 2.0)],
            IMPROVEMENT_EVENT_ID: [(30, 1.0), (50, 2.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 41 'Inverted, last must not be confirmed' failed!"

    # Symmetric mode
    # Unconfirmed vs. next-confirmed vs. sustained
    test_dataframe_no_next_sustained = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50],
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_no_next_sustained,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
            "opt_baseline_type": "fixed",
        },
    ), "Test 42 'unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_no_next_sustained,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 43 'next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_no_next_sustained,
        targets_dict={},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
        },
    ), "Test 44 'sustained' failed!"

    # Test various confirmation durations
    test_dataframe_durations = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [1, 2.5, 2.5, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_durations,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.5)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.5)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.5)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 45 '10 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_durations,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 46 '20 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_durations,
        targets_dict={},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_baseline_type": "fixed",
        },
    ), "Test 47 '30 units confirmed' failed!"

    # Test left-hand side tolerance
    test_dataframe_left_tolerance = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [1, 2.5, 2.5, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_left_tolerance,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 0,
            "opt_baseline_type": "fixed",
        },
    ), "Test 48 '15 units confirmed, no tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_left_tolerance,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.5)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.5)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.5)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 15,
            "opt_confirmation_time_left_side_max_tolerance": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 49 '15 units confirmed, 5 units tolerance' failed!"

    # Test right-hand side max. distance constraint
    test_dataframe_right_constraint = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 30, 40],
            EDSS_SCORE: [1, 2.5, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_right_constraint,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_time_right_side_max_tolerance": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 50 '10 units confirmed, 10 units tolerance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_right_constraint,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_time_right_side_max_tolerance": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 51 '10 units confirmed, 5 units tolerance' failed!"

    # Test minimal distance for sustained
    test_dataframe_sustained_minimal_distance = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [1, 2.5, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_sustained_minimal_distance,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 52 'Sustained, minimum 20 units' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_sustained_minimal_distance,
        targets_dict={},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_confirmation_sustained_minimal_distance": 21,
            "opt_baseline_type": "fixed",
        },
    ), "Test 53 'Sustained, minimum 21 units' failed!"

    # Test all vs. last confirmed
    test_dataframe_all_vs_last = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [1, 2.0, 1.5, 1.5, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_all_vs_last,
        targets_dict={},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "opt_baseline_type": "fixed",
        },
    ), "Test 54 '30 units confirmed, all values' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_all_vs_last,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "last",
            "opt_baseline_type": "fixed",
        },
    ), "Test 55 '30 units confirmed, last only' failed!"

    # Minimum vs. monotonic
    test_dataframe_min_vs_monotonic = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [1, 2.5, 2.0, 2.0, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_min_vs_monotonic,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_type": "minimum",
            "opt_baseline_type": "fixed",
        },
    ), "Test 56 '30 units confirmed, minimum' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_min_vs_monotonic,
        targets_dict={},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_type": "monotonic",
            "opt_baseline_type": "fixed",
        },
    ), "Test 57 '30 units confirmed, monotonic' failed!"

    # Minimum/monotonic - correct event scores?
    test_dataframe_min_vs_monotonic_event_scores = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40],
            EDSS_SCORE: [1, 2.5, 2.5, 3.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_min_vs_monotonic_event_scores,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.5)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.5)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.5)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_type": "minimum",
            "opt_baseline_type": "fixed",
        },
    ), "Test 58 '20 units confirmed, minimum' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_min_vs_monotonic_event_scores,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.5)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.5)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.5)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1.0)],
            ACCRUAL_EVENT_ID: [(10, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_type": "monotonic",
            "opt_baseline_type": "fixed",
        },
    ), "Test 59 '20 units confirmed, monotonic' failed!"

    # No confirmation requirement for last assessment
    test_dataframe_last_confirmed = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50],
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.5, 3.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_last_confirmed,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": True,
            "opt_baseline_type": "fixed",
        },
    ), "Test 60 'Last requires confirmation' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_last_confirmed,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (50, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True)],
            IS_PIRA_REBASELINE: [(30, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0), (50, 3.0)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (50, 3.0)],
            IS_EVENT: [(30, True), (50, True)],
            IS_ACCRUAL_EVENT: [(30, True), (50, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (50, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0), (50, 3.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (50, 2.0)],
            EVENT_ID: [(30, 1.0), (50, 2.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0), (50, 2.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_confirmation_require_confirmation_for_last_visit": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 61 'Last does not require confirmation' failed!"


def test_relapse_independent_baselines():
    # Fixed vs. roving without/with confirmation
    test_dataframe_fixed_roving = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
            EDSS_SCORE: [5.0, 4.0, 4.5, 4.0, 4.0, 4.5, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving,
        targets_dict={},
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
        },
    ), "Test 1 'Fixed baseline' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True), (60, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 4.0), (60, 3.5)],
            IS_PIRA_REBASELINE: [(10, True), (60, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 4.0), (60, 3.5)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_roving_reference_confirmation_time": 0,
        },
    ), "Test 2 'Roving unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True), (30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 4.5), (30, 4.0)],
            IS_PIRA_REBASELINE: [(10, True), (30, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 4.5), (30, 4.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
        },
    ), "Test 3 'Roving next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 4.5)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 4.5)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
        },
    ), "Test 4 'Roving 20 units confirmed' failed!"

    # Roving reference all vs. last confirmed
    test_dataframe_roving_all_vs_last = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
            EDSS_SCORE: [5.0, 4.0, 4.5, 4.0, 4.0, 4.5, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_roving_all_vs_last,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 4.5)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 4.5)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
            "opt_roving_reference_confirmation_included_values": "all",
        },
    ), "Test 5 'Roving reference, 20 units confirmed, all values' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_roving_all_vs_last,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 4.0)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 4.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
            "opt_roving_reference_confirmation_included_values": "last",
        },
    ), "Test 6 'Roving reference, 20 units confirmed, all values' failed!"

    # Roving reference with left- or right-hand tolerance/constraint
    test_dataframe_roving_left_right = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
            EDSS_SCORE: [5.0, 4.0, 4.0, 4.5, 4.0, 4.5, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_roving_left_right,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 4.5)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 4.5)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 15,
            "opt_roving_reference_confirmation_time_left_side_max_tolerance": 0,
            "opt_roving_reference_confirmation_time_right_side_max_tolerance": np.inf,
        },
    ), (
        "Test 7 'Roving reference, 15 units confirmed, no left hand side tolerance' failed!"
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_roving_left_right,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 4.0)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 4.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 15,
            "opt_roving_reference_confirmation_time_left_side_max_tolerance": 5,
            "opt_roving_reference_confirmation_time_right_side_max_tolerance": np.inf,
        },
    ), (
        "Test 8 'Roving reference, 15 units confirmed, 5 units left hand side tolerance' failed!"
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_roving_left_right,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 4.0)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 4.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 5,
            "opt_roving_reference_confirmation_time_left_side_max_tolerance": 0,
            "opt_roving_reference_confirmation_time_right_side_max_tolerance": 5,
        },
    ), (
        "Test 9 'Roving reference, 5 units confirmed, no right hand side constraint' failed!"
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_roving_left_right,
        targets_dict={},
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 5,
            "opt_roving_reference_confirmation_time_left_side_max_tolerance": 0,
            "opt_roving_reference_confirmation_time_right_side_max_tolerance": 4,
        },
    ), (
        "Test 10 'Roving reference, 5 units confirmed, 4 units right hand side constraint' failed!"
    )
    # Inverted mode
    # Fixed vs. roving without/with confirmation
    test_dataframe_fixed_roving_inv = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
            EDSS_SCORE: [1.0, 2.0, 1.5, 2.0, 2.0, 1.5, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving_inv,
        targets_dict={},
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "fixed",
        },
    ), "Test 11 'Fixed baseline' for improvement failed!"
    test_dataframe_fixed_roving_inv = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
            EDSS_SCORE: [1.0, 2.0, 1.5, 2.0, 2.0, 1.5, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving_inv,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True), (60, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0), (60, 2.5)],
            # IS_PIRA_REBASELINE: [(10, True), (60, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0), (60, 2.5)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_roving_reference_confirmation_time": 0,
        },
    ), "Test 12 'Roving reference, unconfirmed' for improvement failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving_inv,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 1.5)],
            # IS_PIRA_REBASELINE: [(10, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 1.5)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
            "opt_roving_reference_confirmation_included_values": "all",
        },
    ), "Test 13 'Roving reference, distance-confirmed, all' for improvement failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving_inv,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # IS_PIRA_REBASELINE: [(10, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 20,
            "opt_roving_reference_confirmation_included_values": "last",
        },
    ), "Test 14 'Roving reference, distance-confirmed, last' for improvement failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving_inv,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True), (30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 1.5), (30, 2.0)],
            # IS_PIRA_REBASELINE: [(10, True), (30, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 1.5), (30, 2.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
        },
    ), "Test 15 'Roving reference, next-confirmed' for improvement failed!"
    test_dataframe_fixed_roving_inv_tol = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
            EDSS_SCORE: [1.0, 2.0, 2.0, 1.5, 2.0, 1.5, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving_inv_tol,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 1.5)],
            # IS_PIRA_REBASELINE: [(10, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 1.5)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 15,
            "opt_roving_reference_confirmation_time_left_side_max_tolerance": 0,
        },
    ), "Test 16 'Roving reference, no left-hand tolerance' for improvement failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving_inv_tol,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # IS_PIRA_REBASELINE: [(10, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 15,
            "opt_roving_reference_confirmation_time_left_side_max_tolerance": 5,
        },
    ), "Test 17 'Roving reference, with left-hand tolerance' for improvement failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving_inv_tol,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # IS_PIRA_REBASELINE: [(10, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 5,
            "opt_roving_reference_confirmation_time_right_side_max_tolerance": np.inf,
        },
    ), "Test 18 'Roving reference, no right-hand constraint' for improvement failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_roving_inv_tol,
        targets_dict={},
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 5,
            "opt_roving_reference_confirmation_time_right_side_max_tolerance": 4,
        },
    ), "Test 18 'Roving reference, with right-hand constraint' for improvement failed!"


def test_min_increase_settings():
    test_dataframe_accrual = pd.DataFrame(
        {
            TIMESTAMP: [i * 10 for i in range(9)],
            EDSS_SCORE: [0.5 * i for i in range(9)],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_accrual,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (20, True),
                (40, True),
                (60, True),
                (80, True),
            ],
            IS_GENERAL_REBASELINE: [(20, True), (40, True), (60, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (20, 1.0),
                (40, 2.0),
                (60, 3.0),
                (80, 4.0),
            ],
            IS_PIRA_REBASELINE: [(20, True), (40, True), (60, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (20, 1.0),
                (40, 2.0),
                (60, 3.0),
                (80, 4.0),
            ],
            IS_EVENT: [(20, True), (40, True), (60, True), (80, True)],
            IS_ACCRUAL_EVENT: [(20, True), (40, True), (60, True), (80, True)],
            EVENT_TYPE: [
                (20, LABEL_PIRA),
                (40, LABEL_PIRA),
                (60, LABEL_PIRA),
                (80, LABEL_PIRA),
            ],
            EVENT_SCORE: [(20, 1.0), (40, 2.0), (60, 3.0), (80, 4.0)],
            EVENT_REFERENCE_SCORE: [(20, 0.0), (40, 1.0), (60, 2.0), (80, 3.0)],
            EVENT_ID: [(20, 1.0), (40, 2.0), (60, 3.0), (80, 4.0)],
            ACCRUAL_EVENT_ID: [(20, 1.0), (40, 2.0), (60, 3.0), (80, 4.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
            "opt_max_score_that_requires_plus_1": 10,
            "opt_larger_increment_from_0": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 1 'Plus 1 irrespective of reference' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_accrual,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (30, True),
                (50, True),
                (70, True),
            ],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 1.5),
                (50, 2.5),
                (70, 3.5),
            ],
            IS_PIRA_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 1.5),
                (50, 2.5),
                (70, 3.5),
            ],
            IS_EVENT: [(30, True), (50, True), (70, True)],
            IS_ACCRUAL_EVENT: [(30, True), (50, True), (70, True)],
            EVENT_TYPE: [
                (30, LABEL_PIRA),
                (50, LABEL_PIRA),
                (70, LABEL_PIRA),
            ],
            EVENT_SCORE: [(30, 1.5), (50, 2.5), (70, 3.5)],
            EVENT_REFERENCE_SCORE: [(30, 0.0), (50, 1.5), (70, 2.5)],
            EVENT_ID: [(30, 1.0), (50, 2.0), (70, 3.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0), (50, 2.0), (70, 3.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
            "opt_max_score_that_requires_plus_1": 10,
            "opt_larger_increment_from_0": True,
            "opt_baseline_type": "fixed",
        },
    ), "Test 2 'Plus 1.5 from 0, plus 1 else' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_accrual,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (30, True),
                (50, True),
                (70, True),
                (80, True),
            ],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 1.5),
                (50, 2.5),
                (70, 3.5),
                (80, 4.0),
            ],
            IS_PIRA_REBASELINE: [(30, True), (50, True), (70, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 1.5),
                (50, 2.5),
                (70, 3.5),
                (80, 4.0),
            ],
            IS_EVENT: [(30, True), (50, True), (70, True), (80, True)],
            IS_ACCRUAL_EVENT: [(30, True), (50, True), (70, True), (80, True)],
            EVENT_TYPE: [
                (30, LABEL_PIRA),
                (50, LABEL_PIRA),
                (70, LABEL_PIRA),
                (80, LABEL_PIRA),
            ],
            EVENT_SCORE: [(30, 1.5), (50, 2.5), (70, 3.5), (80, 4.0)],
            EVENT_REFERENCE_SCORE: [(30, 0.0), (50, 1.5), (70, 2.5), (80, 3.5)],
            EVENT_ID: [(30, 1.0), (50, 2.0), (70, 3.0), (80, 4.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0), (50, 2.0), (70, 3.0), (80, 4.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
            "opt_max_score_that_requires_plus_1": 3.0,
            "opt_larger_increment_from_0": True,
            "opt_baseline_type": "fixed",
        },
    ), (
        "Test 3 'Plus 1.5 from 0, plus 1 for references up to and including 3.0, plus 0.5 else' failed!"
    )
    test_dataframe_accrual_short = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
            EDSS_SCORE: [0, 0.5, 1.0, 1.5, 3.0, 3.5, 4.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_accrual_short,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (30, True),
                (40, True),
                (60, True),
            ],
            IS_GENERAL_REBASELINE: [(30, True), (40, True), (60, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 1.5),
                (40, 3.0),
                (60, 4.0),
            ],
            IS_PIRA_REBASELINE: [(30, True), (40, True), (60, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 1.5),
                (40, 3.0),
                (60, 4.0),
            ],
            IS_EVENT: [(30, True), (40, True), (60, True)],
            IS_ACCRUAL_EVENT: [(30, True), (40, True), (60, True)],
            EVENT_TYPE: [
                (30, LABEL_PIRA),
                (40, LABEL_PIRA),
                (60, LABEL_PIRA),
            ],
            EVENT_SCORE: [(30, 1.5), (40, 3.0), (60, 4.0)],
            EVENT_REFERENCE_SCORE: [(30, 0.0), (40, 1.5), (60, 3.0)],
            EVENT_ID: [(30, 1.0), (40, 2.0), (60, 3.0)],
            ACCRUAL_EVENT_ID: [(30, 1.0), (40, 2.0), (60, 3.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
            "opt_max_score_that_requires_plus_1": 3.0,
            "opt_larger_increment_from_0": True,
            "opt_baseline_type": "fixed",
        },
    ), (
        "Test 4 'Plus 1.5 from 0, plus 1 for references up to and including 3.0, plus 0.5 else' failed!"
    )
    test_dataframe_improvement = pd.DataFrame(
        {
            TIMESTAMP: [i * 10 for i in range(9)],
            EDSS_SCORE: [4.0 - i * 0.5 for i in range(9)],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_improvement,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (20, True),
                (40, True),
                (60, True),
                (80, True),
            ],
            IS_GENERAL_REBASELINE: [(20, True), (40, True), (60, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (20, 3.0),
                (40, 2.0),
                (60, 1.0),
                (80, 0.0),
            ],
            # IS_PIRA_REBASELINE: [(20, True), (40, True), (60, True), (80, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (20, 3.0),
            #    (40, 2.0),
            #    (60, 1.0),
            #    (80, 0.0),
            # ],
            IS_EVENT: [(20, True), (40, True), (60, True), (80, True)],
            IS_IMPROVEMENT_EVENT: [(20, True), (40, True), (60, True), (80, True)],
            EVENT_TYPE: [
                (20, LABEL_IMPROVEMENT),
                (40, LABEL_IMPROVEMENT),
                (60, LABEL_IMPROVEMENT),
                (80, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [(20, 3.0), (40, 2.0), (60, 1.0), (80, 0.0)],
            EVENT_REFERENCE_SCORE: [(20, 4.0), (40, 3.0), (60, 2.0), (80, 1.0)],
            EVENT_ID: [(20, 1.0), (40, 2.0), (60, 3.0), (80, 4.0)],
            IMPROVEMENT_EVENT_ID: [(20, 1.0), (40, 2.0), (60, 3.0), (80, 4.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
            "opt_max_score_that_requires_plus_1": 10,
            "opt_larger_increment_from_0": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 5 'Plus 1 irrespective of reference, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_improvement,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (20, True),
                (40, True),
                (60, True),
            ],
            IS_GENERAL_REBASELINE: [(20, True), (40, True), (60, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (20, 3.0),
                (40, 2.0),
                (60, 1.0),
            ],
            # IS_PIRA_REBASELINE: [(20, True), (40, True), (60, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (20, 3.0),
            #    (40, 2.0),
            #    (60, 1.0),
            # ],
            IS_EVENT: [(20, True), (40, True), (60, True)],
            IS_IMPROVEMENT_EVENT: [(20, True), (40, True), (60, True)],
            EVENT_TYPE: [
                (20, LABEL_IMPROVEMENT),
                (40, LABEL_IMPROVEMENT),
                (60, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [(20, 3.0), (40, 2.0), (60, 1.0)],
            EVENT_REFERENCE_SCORE: [(20, 4.0), (40, 3.0), (60, 2.0)],
            EVENT_ID: [(20, 1.0), (40, 2.0), (60, 3.0)],
            IMPROVEMENT_EVENT_ID: [(20, 1.0), (40, 2.0), (60, 3.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
            "opt_max_score_that_requires_plus_1": 10,
            "opt_larger_increment_from_0": True,
            "opt_baseline_type": "fixed",
        },
    ), "Test 6 'Plus 1.5 from 0, plus 1 else, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_improvement,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (10, True),
                (30, True),
                (50, True),
                (80, True),
            ],
            IS_GENERAL_REBASELINE: [
                (10, True),
                (30, True),
                (50, True),
                (80, True),
            ],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (10, 3.5),
                (30, 2.5),
                (50, 1.5),
                (80, 0.0),
            ],
            # IS_PIRA_REBASELINE: [
            #    (10, True),
            #    (30, True),
            #    (50, True),
            #    (80, True),
            # ],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (10, 3.5),
            #    (30, 2.5),
            #    (50, 1.5),
            #    (80, 0.0),
            # ],
            IS_EVENT: [
                (10, True),
                (30, True),
                (50, True),
                (80, True),
            ],
            IS_IMPROVEMENT_EVENT: [
                (10, True),
                (30, True),
                (50, True),
                (80, True),
            ],
            EVENT_TYPE: [
                (10, LABEL_IMPROVEMENT),
                (30, LABEL_IMPROVEMENT),
                (50, LABEL_IMPROVEMENT),
                (80, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [(10, 3.5), (30, 2.5), (50, 1.5), (80, 0.0)],
            EVENT_REFERENCE_SCORE: [(10, 4.0), (30, 3.5), (50, 2.5), (80, 1.5)],
            EVENT_ID: [(10, 1.0), (30, 2.0), (50, 3.0), (80, 4.0)],
            IMPROVEMENT_EVENT_ID: [(10, 1.0), (30, 2.0), (50, 3.0), (80, 4.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
            "opt_max_score_that_requires_plus_1": 3.0,
            "opt_larger_increment_from_0": True,
            "opt_baseline_type": "fixed",
        },
    ), (
        "Test 7 'Plus 1.5 from 0, plus 1 for references up to and including 3.0, plus 0.5 else, improvement' failed!"
    )


def test_relapse_independent_minimal_distance():
    # Minimal distance to reference, various distances
    test_dataframe_distances_to_reference = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [1, 2.0, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_reference,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1)],
            ACCRUAL_EVENT_ID: [(10, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 10,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 1 '10 units to reference' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_reference,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(20, True)],
            IS_GENERAL_REBASELINE: [(20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
            IS_PIRA_REBASELINE: [(20, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
            IS_EVENT: [(20, True)],
            IS_ACCRUAL_EVENT: [(20, True)],
            EVENT_TYPE: [(20, LABEL_PIRA)],
            EVENT_SCORE: [(20, 2.0)],
            EVENT_REFERENCE_SCORE: [(20, 1.0)],
            EVENT_ID: [(20, 1)],
            ACCRUAL_EVENT_ID: [(20, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 20,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 2 '20 units to reference' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_reference,
        targets_dict={},
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 31,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 3 '31 units to reference' failed!"

    # Minimal distance to reference with confirmation
    test_dataframe_distances_to_reference_confirmed = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [1, 2.0, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_reference_confirmed,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(20, True)],
            IS_GENERAL_REBASELINE: [(20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
            IS_PIRA_REBASELINE: [(20, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
            IS_EVENT: [(20, True)],
            IS_ACCRUAL_EVENT: [(20, True)],
            EVENT_TYPE: [(20, LABEL_PIRA)],
            EVENT_SCORE: [(20, 2.0)],
            EVENT_REFERENCE_SCORE: [(20, 1.0)],
            EVENT_ID: [(20, 1)],
            ACCRUAL_EVENT_ID: [(20, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 20,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
        },
    ), "Test 4 '20 units to reference, unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_reference_confirmed,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(20, True)],
            IS_GENERAL_REBASELINE: [(20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
            IS_PIRA_REBASELINE: [(20, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
            IS_EVENT: [(20, True)],
            IS_ACCRUAL_EVENT: [(20, True)],
            EVENT_TYPE: [(20, LABEL_PIRA)],
            EVENT_SCORE: [(20, 2.0)],
            EVENT_REFERENCE_SCORE: [(20, 1.0)],
            EVENT_ID: [(20, 1)],
            ACCRUAL_EVENT_ID: [(20, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 20,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
        },
    ), "Test 5 '20 units to reference, 10 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_reference_confirmed,
        targets_dict={},
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
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
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [1, 2.0, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_previous,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            IS_PIRA_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_ACCRUAL_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_PIRA)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 1.0)],
            EVENT_ID: [(10, 1)],
            ACCRUAL_EVENT_ID: [(10, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 10,
            "opt_minimal_distance_type": "previous",
        },
    ), "Test 7 '10 units to previous' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_previous,
        targets_dict={},
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 11,
            "opt_minimal_distance_type": "previous",
        },
    ), "Test 8 '11 units to previous' failed!"

    # Backtracking
    test_dataframe_backtracking = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [3.5, 3.0, 2.5, 4.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtracking,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True), (20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 3.0), (20, 2.5)],
            IS_PIRA_REBASELINE: [(10, True), (20, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 3.0), (20, 2.5)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": False,
        },
    ), "Test 9 '15 units to reference, without backtracking' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtracking,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(10, True), (20, True), (30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (10, 3.0),
                (20, 2.5),
                (30, 4.0),
            ],
            IS_PIRA_REBASELINE: [(10, True), (20, True), (30, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (10, 3.0),
                (20, 2.5),
                (30, 4.0),
            ],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 4.0)],
            EVENT_REFERENCE_SCORE: [(30, 3.0)],
            EVENT_ID: [(30, 1)],
            ACCRUAL_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 10 '15 units to reference, with backtracking' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtracking,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True), (20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 3.0), (20, 2.5)],
            IS_PIRA_REBASELINE: [(10, True), (20, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 3.0), (20, 2.5)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 25,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 11 '25 units to reference, with backtracking' failed!"

    # Backtracking with confirmation
    test_dataframe_backtracking_confirmed = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
            EDSS_SCORE: [3.5, 3.0, 2.5, 4.5, 4.5, 4.0, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtracking_confirmed,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [
                (10, True),
                (20, True),
                (30, True),
                (50, True),
                (60, True),
            ],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (10, 3.0),
                (20, 2.5),
                (30, 4.5),
                (50, 4.0),
                (60, 3.5),
            ],
            IS_PIRA_REBASELINE: [
                (10, True),
                (20, True),
                (30, True),
                (50, True),
                (60, True),
            ],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (10, 3.0),
                (20, 2.5),
                (30, 4.5),
                (50, 4.0),
                (60, 3.5),
            ],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 4.5)],
            EVENT_REFERENCE_SCORE: [(30, 3.0)],
            EVENT_ID: [(30, 1)],
            ACCRUAL_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
        },
    ), "Test 12 '15 units to reference, with backtracking, 10 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtracking_confirmed,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [
                (10, True),
                (20, True),
                (30, True),
                (60, True),
            ],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (10, 3.0),
                (20, 2.5),
                (30, 4.0),
                (60, 3.5),
            ],
            IS_PIRA_REBASELINE: [
                (10, True),
                (20, True),
                (30, True),
                (60, True),
            ],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (10, 3.0),
                (20, 2.5),
                (30, 4.0),
                (60, 3.5),
            ],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 4.0)],
            EVENT_REFERENCE_SCORE: [(30, 3.0)],
            EVENT_ID: [(30, 1)],
            ACCRUAL_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
        },
    ), "Test 13 '15 units to reference, with backtracking, 20 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtracking_confirmed,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(10, True), (20, True), (30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (10, 3.0),
                (20, 2.5),
                (30, 3.5),
            ],
            IS_PIRA_REBASELINE: [(10, True), (20, True), (30, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (10, 3.0),
                (20, 2.5),
                (30, 3.5),
            ],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 3.5)],
            EVENT_REFERENCE_SCORE: [(30, 2.5)],
            EVENT_ID: [(30, 1)],
            ACCRUAL_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 0,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
        },
    ), "Test 14 'No minimal distance, 30 units confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtracking_confirmed,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True), (20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 3.0), (20, 2.5)],
            IS_PIRA_REBASELINE: [(10, True), (20, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 3.0), (20, 2.5)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
        },
    ), "Test 15 '15 units to reference, with backtracking, 30 units confirmed' failed!"
    # Inverted mode
    test_dataframe_distances_to_previous_inv = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [3.0, 2.0, 2.0, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_previous_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # IS_PIRA_REBASELINE: [(10, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1)],
            IMPROVEMENT_EVENT_ID: [(10, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 10,
            "opt_minimal_distance_type": "previous",
        },
    ), "Test 16 '10 units to previous, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_previous_inv,
        targets_dict={},
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 11,
            "opt_minimal_distance_type": "previous",
        },
    ), "Test 17 '11 units to previous, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_previous_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True)],
            IS_GENERAL_REBASELINE: [(10, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 2.0)],
            # IS_PIRA_REBASELINE: [(10, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 2.0)],
            IS_EVENT: [(10, True)],
            IS_IMPROVEMENT_EVENT: [(10, True)],
            EVENT_TYPE: [(10, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(10, 2.0)],
            EVENT_REFERENCE_SCORE: [(10, 3.0)],
            EVENT_ID: [(10, 1)],
            IMPROVEMENT_EVENT_ID: [(10, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 10,
            "opt_minimal_distance_type": "reference",
        },
    ), "Test 18 '10 units to reference, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_previous_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(20, True)],
            IS_GENERAL_REBASELINE: [(20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
            # IS_PIRA_REBASELINE: [(20, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
            IS_EVENT: [(20, True)],
            IS_IMPROVEMENT_EVENT: [(20, True)],
            EVENT_TYPE: [(20, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(20, 2.0)],
            EVENT_REFERENCE_SCORE: [(20, 3.0)],
            EVENT_ID: [(20, 1)],
            IMPROVEMENT_EVENT_ID: [(20, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 20,
            "opt_minimal_distance_type": "reference",
        },
    ), "Test 19 '20 units to reference, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_previous_inv,
        targets_dict={},
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 31,
            "opt_minimal_distance_type": "reference",
        },
    ), "Test 20 '31 units to reference, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_previous_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(20, True)],
            IS_GENERAL_REBASELINE: [(20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
            # IS_PIRA_REBASELINE: [(20, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
            IS_EVENT: [(20, True)],
            IS_IMPROVEMENT_EVENT: [(20, True)],
            EVENT_TYPE: [(20, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(20, 2.0)],
            EVENT_REFERENCE_SCORE: [(20, 3.0)],
            EVENT_ID: [(20, 1)],
            IMPROVEMENT_EVENT_ID: [(20, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 20,
            "opt_minimal_distance_type": "reference",
        },
    ), "Test 21 '20 units to reference, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_previous_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(20, True)],
            IS_GENERAL_REBASELINE: [(20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
            # IS_PIRA_REBASELINE: [(20, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
            IS_EVENT: [(20, True)],
            IS_IMPROVEMENT_EVENT: [(20, True)],
            EVENT_TYPE: [(20, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(20, 2.0)],
            EVENT_REFERENCE_SCORE: [(20, 3.0)],
            EVENT_ID: [(20, 1)],
            IMPROVEMENT_EVENT_ID: [(20, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 20,
            "opt_minimal_distance_type": "reference",
        },
    ), "Test 22 '20 units to reference, 10 units confirmed, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_distances_to_previous_inv,
        targets_dict={},
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 11,
            "opt_baseline_type": "fixed",
            "opt_minimal_distance_time": 20,
            "opt_minimal_distance_type": "reference",
        },
    ), "Test 23 '20 units to reference, 11 units confirmed, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pd.DataFrame(
            {
                TIMESTAMP: [0, 10, 20, 30],
                EDSS_SCORE: [2.5, 3.0, 3.5, 2.0],
            }
        ),
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True), (20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 3.0), (20, 3.5)],
            # IS_PIRA_REBASELINE: [(10, True), (20, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 3.0), (20, 3.5)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": False,
        },
    ), "Test 24 'No backtracking, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pd.DataFrame(
            {
                TIMESTAMP: [0, 10, 20, 30],
                EDSS_SCORE: [2.5, 3.0, 3.0, 2.0],
            }
        ),
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(10, True), (30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 3.0), (30, 2.0)],
            # IS_PIRA_REBASELINE: [(30, True)],  # (10, True),
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],  # (10, 3.0),
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 3.0)],
            EVENT_ID: [(30, 1)],
            IMPROVEMENT_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": False,
        },
    ), "Test 25 'No backtracking, improvement' failed!"
    test_dataframe_backtrack_inv = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30],
            EDSS_SCORE: [2.5, 3.0, 3.5, 2.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtrack_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(10, True), (20, True), (30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (10, 3.0),
                (20, 3.5),
                (30, 2.0),
            ],
            # IS_PIRA_REBASELINE: [(30, True)],  # (10, True), (20, True),
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 2.0),
            # ],  # (10, 3.0), (20, 3.5),
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 3.0)],
            EVENT_ID: [(30, 1)],
            IMPROVEMENT_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 26 'With backtracking, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtrack_inv,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True), (20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (10, 3.0),
                (20, 3.5),
            ],
            # IS_PIRA_REBASELINE: [(10, True), (20, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (10, 3.0),
            #    (20, 3.5),
            # ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 25,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 27 'With backtracking, improvement' failed!"
    test_dataframe_backtrack_confirm_inv = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
            EDSS_SCORE: [2.5, 3.0, 3.5, 1.5, 1.5, 2.0, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtrack_confirm_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(10, True), (20, True), (30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (10, 3.0),
                (20, 3.5),
                (30, 2.5),
            ],
            # IS_PIRA_REBASELINE: [(30, True)],  # (10, True), (20, True),
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 2.5),
            # ],  # (10, 3.0),(20, 3.5),
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 2.5)],
            EVENT_REFERENCE_SCORE: [(30, 3.5)],
            EVENT_ID: [(30, 1)],
            IMPROVEMENT_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 10,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 28 'With backtracking, confirmed, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtrack_confirm_inv,
        targets_dict={
            IS_GENERAL_REBASELINE: [(10, True), (20, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (10, 3.0),
                (20, 3.5),
            ],
            # IS_PIRA_REBASELINE: [(10, True), (20, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (10, 3.0),
            #    (20, 3.5),
            # ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_minimal_distance_time": 15,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 29 'With backtracking, confirmed, improvement' failed!"


def test_relapse_independent_first_vs_all_events():
    test_dataframe_first_all_events = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60, 70],
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_first_all_events,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (50, True), (70, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 2.0),
                (50, 3.0),
                (70, 4.0),
            ],
            IS_PIRA_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 2.0),
                (50, 3.0),
                (70, 4.0),
            ],
            IS_EVENT: [(30, True), (50, True), (70, True)],
            IS_ACCRUAL_EVENT: [(30, True), (50, True), (70, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (50, LABEL_PIRA), (70, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0), (50, 3.0), (70, 4.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (50, 2.0), (70, 3.0)],
            EVENT_ID: [(30, 1), (50, 2), (70, 3)],
            ACCRUAL_EVENT_ID: [(30, 1), (50, 2), (70, 3)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "return_first_event_only": False,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 1 'Return all events' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_first_all_events,
        targets_dict={
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1)],
            ACCRUAL_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "return_first_event_only": True,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 2 'Return first event only' failed!"


def test_relapse_independent_multiple_events_rebaselining():
    # Accrual/improvement vs. symmetric with fixed
    test_dataframe_fixed_annotation_modes = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            EDSS_SCORE: [1, 1, 1.5, 2.5, 2.0, 1.5, 1.5, 2.5, 3.0, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_annotation_modes,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (90, True)],
            IS_GENERAL_REBASELINE: [(30, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.5), (90, 3.5)],
            IS_PIRA_REBASELINE: [(30, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.5), (90, 3.5)],
            IS_EVENT: [(30, True), (90, True)],
            IS_ACCRUAL_EVENT: [(30, True), (90, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (90, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.5), (90, 3.5)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (90, 2.5)],
            EVENT_ID: [(30, 1), (90, 2)],
            ACCRUAL_EVENT_ID: [(30, 1), (90, 2)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 0a 'Fixed baseline, accrual, unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_annotation_modes,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
            IS_GENERAL_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0), (80, 3.0)],
            IS_PIRA_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (80, 3.0)],
            IS_EVENT: [(30, True), (80, True)],
            IS_ACCRUAL_EVENT: [(30, True), (80, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0), (80, 3.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 2.0)],
            EVENT_ID: [(30, 1), (80, 2)],
            ACCRUAL_EVENT_ID: [(30, 1), (80, 2)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 0b 'Fixed baseline, accrual, next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_annotation_modes,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (50, True), (70, True), (90, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 2.5),
                (50, 1.5),
                (70, 2.5),
                (90, 3.5),
            ],
            IS_PIRA_REBASELINE: [(30, True), (50, True), (70, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 2.5),
                (50, 1.5),
                (70, 2.5),
                (90, 3.5),
            ],
            IS_EVENT: [(30, True), (50, True), (70, True), (90, True)],
            IS_ACCRUAL_EVENT: [(30, True), (70, True), (90, True)],
            IS_IMPROVEMENT_EVENT: [(50, True)],
            EVENT_TYPE: [
                (30, LABEL_PIRA),
                (50, LABEL_IMPROVEMENT),
                (70, LABEL_PIRA),
                (90, LABEL_PIRA),
            ],
            EVENT_SCORE: [(30, 2.5), (50, 1.5), (70, 2.5), (90, 3.5)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (50, 2.5), (70, 1.5), (90, 2.5)],
            EVENT_ID: [(30, 1), (50, 2), (70, 3), (90, 4)],
            ACCRUAL_EVENT_ID: [(30, 1), (70, 2), (90, 3)],
            IMPROVEMENT_EVENT_ID: [(50, 1)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 0c 'Fixed baseline, symmetric, unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_annotation_modes,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
            IS_GENERAL_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0), (80, 3.0)],
            IS_PIRA_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (80, 3.0)],
            IS_EVENT: [(30, True), (80, True)],
            IS_ACCRUAL_EVENT: [(30, True), (80, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0), (80, 3.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 2.0)],
            EVENT_ID: [(30, 1), (80, 2)],
            ACCRUAL_EVENT_ID: [(30, 1), (80, 2)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 0d 'Fixed baseline, symmetric, next-confirmed' failed!"
    test_dataframe_fixed_annotation_modes_2 = pd.DataFrame(
        {
            "days_after_baseline": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "edss_score": [1, 1, 1.5, 2.5, 2.0, 1.5, 1.5, 2.5, 3.0, 3.5],
        }
    )
    test_dataframe_fixed_annotation_modes_2["edss_score"] = (
        6.0 - test_dataframe_fixed_annotation_modes_2["edss_score"]
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_annotation_modes_2,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (90, True)],
            IS_GENERAL_REBASELINE: [(30, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (90, 2.5)],
            # IS_PIRA_REBASELINE: [(30, True), (90, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.5), (90, 3.5)],
            IS_EVENT: [(30, True), (90, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (90, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT), (90, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 3.5), (90, 2.5)],
            EVENT_REFERENCE_SCORE: [(30, 5.0), (90, 3.5)],
            EVENT_ID: [(30, 1), (90, 2)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (90, 2)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 0e 'Fixed baseline, accrual, unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_annotation_modes_2,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
            IS_GENERAL_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 4.0), (80, 3.0)],
            # IS_PIRA_REBASELINE: [(30, True), (80, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 4.0), (80, 3.0)],
            IS_EVENT: [(30, True), (80, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (80, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT), (80, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 4.0), (80, 3.0)],
            EVENT_REFERENCE_SCORE: [(30, 5.0), (80, 4.0)],
            EVENT_ID: [(30, 1), (80, 2)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (80, 2)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 0f 'Fixed baseline, accrual, next-confirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_annotation_modes_2,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (50, True), (70, True), (90, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 3.5),
                (50, 4.5),
                (70, 3.5),
                (90, 2.5),
            ],
            IS_PIRA_REBASELINE: [(30, True), (50, True), (70, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 3.5),
                (50, 4.5),
                (70, 3.5),
                (90, 2.5),
            ],
            IS_EVENT: [(30, True), (50, True), (70, True), (90, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (70, True), (90, True)],
            IS_ACCRUAL_EVENT: [(50, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (50, LABEL_PIRA),
                (70, LABEL_IMPROVEMENT),
                (90, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [(30, 3.5), (50, 4.5), (70, 3.5), (90, 2.5)],
            EVENT_REFERENCE_SCORE: [(30, 5.0), (50, 3.5), (70, 4.5), (90, 3.5)],
            EVENT_ID: [(30, 1), (50, 2), (70, 3), (90, 4)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (70, 2), (90, 3)],
            ACCRUAL_EVENT_ID: [(50, 1)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 0g 'Fixed baseline, symmetric, unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_fixed_annotation_modes_2,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
            IS_GENERAL_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 4.0), (80, 3.0)],
            IS_PIRA_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 4.0), (80, 3.0)],
            IS_EVENT: [(30, True), (80, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (80, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT), (80, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 4.0), (80, 3.0)],
            EVENT_REFERENCE_SCORE: [(30, 5.0), (80, 4.0)],
            EVENT_ID: [(30, 1), (80, 2)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (80, 2)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 0h 'Fixed baseline, symmetric, next-confirmed' failed!"

    # Fixed vs. roving
    test_dataframe_multiple_fixed_roving = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60, 70],
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.0, 1.5, 1.5, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_fixed_roving,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1)],
            ACCRUAL_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
            "opt_roving_reference_require_confirmation": False,
        },
    ), "Test 1 'Fixed baseline' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_fixed_roving,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (70, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 2.0),
                (50, 1.5),
                (70, 2.5),
            ],
            IS_PIRA_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 2.0),
                (50, 1.5),
                (70, 2.5),
            ],
            IS_EVENT: [(30, True), (70, True)],
            IS_ACCRUAL_EVENT: [(30, True), (70, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (70, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0), (70, 2.5)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (70, 1.5)],
            EVENT_ID: [(30, 1), (70, 2)],
            ACCRUAL_EVENT_ID: [(30, 1), (70, 2)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
        },
    ), "Test 2 'Roving reference' failed!"

    # With and without confirmation
    test_dataframe_multiple_confirmation = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            EDSS_SCORE: [1, 1, 1.5, 2.5, 2.0, 1.5, 1.5, 2.5, 3.0, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_confirmation,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (90, True)],
            IS_GENERAL_REBASELINE: [(30, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.5), (90, 3.5)],
            IS_PIRA_REBASELINE: [(30, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.5), (90, 3.5)],
            IS_EVENT: [(30, True), (90, True)],
            IS_ACCRUAL_EVENT: [(30, True), (90, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (90, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.5), (90, 3.5)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (90, 2.5)],
            EVENT_ID: [(30, 1), (90, 2)],
            ACCRUAL_EVENT_ID: [(30, 1), (90, 2)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
        },
    ), "Test 3 'Fixed baseline, progression unconfirmed' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_confirmation,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
            IS_GENERAL_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0), (80, 3.0)],
            IS_PIRA_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (80, 3.0)],
            IS_EVENT: [(30, True), (80, True)],
            IS_ACCRUAL_EVENT: [(30, True), (80, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0), (80, 3.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 2.0)],
            EVENT_ID: [(30, 1), (80, 2)],
            ACCRUAL_EVENT_ID: [(30, 1), (80, 2)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 4 'Fixed baseline, progression next-confirmed' failed!"

    # Symmetric
    test_dataframe_multiple_symmetric = pd.DataFrame(
        {
            TIMESTAMP: [i * 10 for i in range(10)],
            EDSS_SCORE: [1, 1, 1.5, 2.5, 2.0, 1.5, 1.5, 2.5, 3.0, 3.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_symmetric,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (30, True),
                (50, True),
                (70, True),
                (90, True),
            ],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 2.5),
                (50, 1.5),
                (70, 2.5),
                (90, 3.5),
            ],
            IS_PIRA_REBASELINE: [(30, True), (50, True), (70, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 2.5),
                (50, 1.5),
                (70, 2.5),
                (90, 3.5),
            ],
            IS_EVENT: [(30, True), (50, True), (70, True), (90, True)],
            IS_ACCRUAL_EVENT: [(30, True), (70, True), (90, True)],
            IS_IMPROVEMENT_EVENT: [(50, True)],
            EVENT_TYPE: [
                (30, LABEL_PIRA),
                (50, LABEL_IMPROVEMENT),
                (70, LABEL_PIRA),
                (90, LABEL_PIRA),
            ],
            EVENT_SCORE: [
                (30, 2.5),
                (50, 1.5),
                (70, 2.5),
                (90, 3.5),
            ],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (50, 2.5), (70, 1.5), (90, 2.5)],
            EVENT_ID: [(30, 1), (50, 2), (70, 3), (90, 4)],
            ACCRUAL_EVENT_ID: [(30, 1), (70, 2), (90, 3)],
            IMPROVEMENT_EVENT_ID: [(50, 1)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 5 'Fixed baseline, unconfirmed, symmetric' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_symmetric,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (30, True),
                (80, True),
            ],
            IS_GENERAL_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 2.0),
                (80, 3.0),
            ],
            IS_PIRA_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 2.0),
                (80, 3.0),
            ],
            IS_EVENT: [(30, True), (80, True)],
            IS_ACCRUAL_EVENT: [(30, True), (80, True)],
            EVENT_TYPE: [
                (30, LABEL_PIRA),
                (80, LABEL_PIRA),
            ],
            EVENT_SCORE: [
                (30, 2.0),
                (80, 3.0),
            ],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 2.0)],
            EVENT_ID: [(30, 1), (80, 2)],
            ACCRUAL_EVENT_ID: [(30, 1), (80, 2)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 6 'Fixed baseline, next-confirmed, symmetric' failed!"
    # Symmetric and improvement
    test_dataframe_multiple_symm_inv = pd.DataFrame(
        {
            TIMESTAMP: [i * 10 for i in range(10)],
            EDSS_SCORE: [5.0, 5.0, 4.5, 3.5, 4.0, 4.5, 4.5, 3.5, 3.0, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_symm_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (30, True),
                (90, True),
            ],
            IS_GENERAL_REBASELINE: [(30, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 3.5),
                (90, 2.5),
            ],
            # IS_PIRA_REBASELINE: [(30, True), (90, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 3.5),
            #    (90, 2.5),
            # ],
            IS_EVENT: [(30, True), (90, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (90, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (90, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 3.5),
                (90, 2.5),
            ],
            EVENT_REFERENCE_SCORE: [(30, 5.0), (90, 3.5)],
            EVENT_ID: [(30, 1), (90, 2)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (90, 2)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 7 'Fixed baseline, unconfirmed, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_symm_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (30, True),
                (80, True),
            ],
            IS_GENERAL_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 4.0),
                (80, 3.0),
            ],
            # IS_PIRA_REBASELINE: [(30, True), (80, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 4.0),
            #    (80, 3.0),
            # ],
            IS_EVENT: [(30, True), (80, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (80, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (80, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 4.0),
                (80, 3.0),
            ],
            EVENT_REFERENCE_SCORE: [(30, 5.0), (80, 4.0)],
            EVENT_ID: [(30, 1), (80, 2)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (80, 2)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 8 'Fixed baseline, next-confirmed, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_symm_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (30, True),
                (50, True),
                (70, True),
                (90, True),
            ],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 3.5),
                (50, 4.5),
                (70, 3.5),
                (90, 2.5),
            ],
            IS_PIRA_REBASELINE: [(30, True), (50, True), (70, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 3.5),
                (50, 4.5),
                (70, 3.5),
                (90, 2.5),
            ],
            IS_EVENT: [(30, True), (50, True), (70, True), (90, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (70, True), (90, True)],
            IS_ACCRUAL_EVENT: [(50, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (50, LABEL_PIRA),
                (70, LABEL_IMPROVEMENT),
                (90, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 3.5),
                (50, 4.5),
                (70, 3.5),
                (90, 2.5),
            ],
            EVENT_REFERENCE_SCORE: [(30, 5.0), (50, 3.5), (70, 4.5), (90, 3.5)],
            EVENT_ID: [(30, 1), (50, 2), (70, 3), (90, 4)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (70, 2), (90, 3)],
            ACCRUAL_EVENT_ID: [(50, 1)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 9 'Fixed baseline, unconfirmed, symmetric' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_symm_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (30, True),
                (80, True),
            ],
            IS_GENERAL_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 4.0),
                (80, 3.0),
            ],
            IS_PIRA_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 4.0),
                (80, 3.0),
            ],
            IS_EVENT: [(30, True), (80, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (80, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (80, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 4.0),
                (80, 3.0),
            ],
            EVENT_REFERENCE_SCORE: [(30, 5.0), (80, 4.0)],
            EVENT_ID: [(30, 1), (80, 2)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (80, 2)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 10 'Fixed baseline, next-confirmed, symmetric' failed!"

    # Multi-event and roving
    test_dataframe_multiple_rov = pd.DataFrame(
        {
            TIMESTAMP: [i * 10 for i in range(8)],
            EDSS_SCORE: [1.0, 1.0, 1.5, 2.0, 2.0, 1.5, 1.5, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_rov,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            IS_PIRA_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1)],
            ACCRUAL_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 11 'Fixed baseline' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_rov,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (70, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 2.0),
                (50, 1.5),
                (70, 2.5),
            ],
            IS_PIRA_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 2.0),
                (50, 1.5),
                (70, 2.5),
            ],
            IS_EVENT: [(30, True), (70, True)],
            IS_ACCRUAL_EVENT: [(30, True), (70, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (70, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0), (70, 2.5)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (70, 1.5)],
            EVENT_ID: [(30, 1), (70, 2)],
            ACCRUAL_EVENT_ID: [(30, 1), (70, 2)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
        },
    ), "Test 12 'Roving baseline' failed!"
    test_dataframe_multiple_rov[EDSS_SCORE] = (
        4.0 - test_dataframe_multiple_rov[EDSS_SCORE]
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_rov,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            # IS_PIRA_REBASELINE: [(30, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 3.0)],
            EVENT_ID: [(30, 1)],
            IMPROVEMENT_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 13 'Fixed baseline, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_multiple_rov,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (70, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 2.0),
                (50, 2.5),
                (70, 1.5),
            ],
            # IS_PIRA_REBASELINE: [(30, True), (70, True)],  # , (50, True)
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 2.0),
            #    (70, 1.5),
            # ],  # (50, 2.5),
            IS_EVENT: [(30, True), (70, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (70, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT), (70, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 2.0), (70, 1.5)],
            EVENT_REFERENCE_SCORE: [(30, 3.0), (70, 2.5)],
            EVENT_ID: [(30, 1), (70, 2)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (70, 2)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
        },
    ), "Test 14 'Roving baseline, improvement' failed!"

    # With backtracking and minimal distance
    test_dataframe_backtrack = pd.DataFrame(
        {
            TIMESTAMP: [i * 10 for i in range(9)],
            EDSS_SCORE: [1.0, 1.0, 1.5, 2.0, 2.0, 0.5, 0.5, 2.5, 2.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtrack,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (70, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 2.0),
                (50, 0.5),
                (70, 2.5),
            ],
            IS_PIRA_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 2.0),
                (50, 0.5),
                (70, 2.5),
            ],
            IS_EVENT: [(30, True), (70, True)],
            IS_ACCRUAL_EVENT: [(30, True), (70, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (70, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0), (70, 2.5)],
            EVENT_REFERENCE_SCORE: [(30, 1.0), (70, 0.5)],
            EVENT_ID: [(30, 1), (70, 2)],
            ACCRUAL_EVENT_ID: [(30, 1), (70, 2)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
        },
    ), "Test 15 'Next-confirmed, no minimal distance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtrack,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 2.0),
                (50, 0.5),
            ],
            IS_PIRA_REBASELINE: [(30, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (30, 2.0),
                (50, 0.5),
            ],
            IS_EVENT: [(30, True)],
            IS_ACCRUAL_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_PIRA)],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 1.0)],
            EVENT_ID: [(30, 1)],
            ACCRUAL_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
            "opt_minimal_distance_time": 30,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 16 'Next-confirmed, with distance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pd.DataFrame(
            {
                TIMESTAMP: [i * 10 for i in range(9)],
                EDSS_SCORE: [1.0, 1.0, 1.5, 1.5, 1.5, 0.5, 0.5, 2.5, 2.5],
            }
        ),
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(70, True)],
            IS_GENERAL_REBASELINE: [(70, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (70, 2.5),
                (50, 0.5),
            ],
            IS_PIRA_REBASELINE: [(70, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (70, 2.5),
                (50, 0.5),
            ],
            IS_EVENT: [(70, True)],
            IS_ACCRUAL_EVENT: [(70, True)],
            EVENT_TYPE: [(70, LABEL_PIRA)],
            EVENT_SCORE: [(70, 2.5)],
            EVENT_REFERENCE_SCORE: [(70, 1.0)],
            EVENT_ID: [(70, 1)],
            ACCRUAL_EVENT_ID: [(70, 1)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
            "opt_minimal_distance_time": 30,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 16b 'Next-confirmed, with distance' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pd.DataFrame(
            {
                TIMESTAMP: [i * 10 for i in range(9)],
                EDSS_SCORE: [1.0, 1.0, 1.5, 2.0, 2.0, 0.5, 0.5, 3.0, 3.0],
            }
        ),
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (70, True)],
            IS_GENERAL_REBASELINE: [(30, True), (70, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (70, 3.0),
                (50, 0.5),
                (30, 2.0),
            ],
            IS_PIRA_REBASELINE: [(30, True), (70, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (70, 3.0),
                (50, 0.5),
                (30, 2.0),
            ],
            IS_EVENT: [(30, True), (70, True)],
            IS_ACCRUAL_EVENT: [(30, True), (70, True)],
            EVENT_TYPE: [(30, LABEL_PIRA), (70, LABEL_PIRA)],
            EVENT_SCORE: [(70, 3.0), (30, 2.0)],
            EVENT_REFERENCE_SCORE: [(70, 2.0), (30, 1)],
            EVENT_ID: [(30, 1), (70, 2)],
            ACCRUAL_EVENT_ID: [(30, 1), (70, 2)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
            "opt_minimal_distance_time": 30,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 16c 'Next-confirmed, with distance' failed!"

    test_dataframe_backtrack[EDSS_SCORE] = 5.0 - test_dataframe_backtrack[EDSS_SCORE]
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtrack,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (70, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 3.0),
                (50, 4.5),
                (70, 2.5),
            ],
            # IS_PIRA_REBASELINE: [(30, True), (70, True)],  # , (50, True)
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 3.0),
            #    (70, 2.5),
            # ],  # (50, 4.5),
            IS_EVENT: [(30, True), (70, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (70, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT), (70, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 3.0), (70, 2.5)],
            EVENT_REFERENCE_SCORE: [(30, 4.0), (70, 4.5)],
            EVENT_ID: [(30, 1), (70, 2)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (70, 2)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
        },
    ), "Test 17 'Next-confirmed, no minimal distance, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_backtrack,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 3.0),
                (50, 4.5),
            ],
            # IS_PIRA_REBASELINE: [(30, True)],  # , (50, True)
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 3.0),
            # ],  # (50, 4.5),
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(30, 3.0)],
            EVENT_REFERENCE_SCORE: [(30, 4.0)],
            EVENT_ID: [(30, 1)],
            IMPROVEMENT_EVENT_ID: [(30, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
            "opt_minimal_distance_time": 30,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 18 'Next-confirmed, with minimal distance, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pd.DataFrame(
            {
                TIMESTAMP: [i * 10 for i in range(9)],
                EDSS_SCORE: [4.0, 4.0, 3.5, 3.5, 3.5, 4.5, 4.5, 2.5, 2.5],
            }
        ),
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(70, True)],
            IS_GENERAL_REBASELINE: [(70, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (70, 2.5),
                (50, 4.5),
            ],
            # IS_PIRA_REBASELINE: [(70, True)],  # , (50, True)
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (70, 2.5),
            # ],  # (50, 4.5),
            IS_EVENT: [(70, True)],
            IS_IMPROVEMENT_EVENT: [(70, True)],
            EVENT_TYPE: [(70, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(70, 2.5)],
            EVENT_REFERENCE_SCORE: [(70, 4.0)],
            EVENT_ID: [(70, 1)],
            IMPROVEMENT_EVENT_ID: [(70, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
            "opt_minimal_distance_time": 30,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 18b 'Next-confirmed, with minimal distance, improvement' failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pd.DataFrame(
            {
                TIMESTAMP: [i * 10 for i in range(9)],
                EDSS_SCORE: [4.0, 4.0, 3.5, 3.0, 3.0, 4.5, 4.5, 2.0, 2.0],
            }
        ),
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(70, True), (30, True)],
            IS_GENERAL_REBASELINE: [(70, True), (50, True), (30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (70, 2.0),
                (50, 4.5),
                (30, 3.0),
            ],
            # IS_PIRA_REBASELINE: [(70, True), (30, True)],  # , (50, True)
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (70, 2.0),
            #    (30, 3.0),
            # ],  # (50, 4.5),
            IS_EVENT: [(70, True), (30, True)],
            IS_IMPROVEMENT_EVENT: [(70, True), (30, True)],
            EVENT_TYPE: [(30, LABEL_IMPROVEMENT), (70, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(70, 2.0), (30, 3.0)],
            EVENT_REFERENCE_SCORE: [(70, 3.0), (30, 4.0)],
            EVENT_ID: [(30, 1), (70, 2)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (70, 2)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
            "opt_minimal_distance_time": 30,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 18c 'Next-confirmed, with minimal distance, improvement' failed!"


def test_relapse_independent_multiple_events_merging():
    # Test case 1 - unconfirmed merged
    test_dataframe_case_1 = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60, 65, 70, 80, 90],
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.5, 4.0],
        }
    )
    test_case_1_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True)],
        IS_GENERAL_REBASELINE: [(30, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 4.5)],
        IS_PIRA_REBASELINE: [(30, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 4.5)],
        IS_EVENT: [(30, True)],
        IS_ACCRUAL_EVENT: [(30, True)],
        EVENT_TYPE: [(30, LABEL_PIRA)],
        EVENT_SCORE: [(30, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0)],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (65, 1),
            (70, 1),
            (80, 1),
        ],
        ACCRUAL_EVENT_ID: [
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
        targets_dict=test_case_1_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 1 failed!"

    # Test case 1b - unconfirmed merged, first only
    test_case_1b_targets = {
        IS_POST_EVENT_REBASELINE: [],
        IS_GENERAL_REBASELINE: [],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [],
        IS_EVENT: [(30, True)],
        IS_ACCRUAL_EVENT: [(30, True)],
        EVENT_TYPE: [(30, LABEL_PIRA)],
        EVENT_SCORE: [(30, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0)],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (65, 1),
            (70, 1),
            (80, 1),
        ],
        ACCRUAL_EVENT_ID: [
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
        targets_dict=test_case_1b_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "return_first_event_only": True,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 1b failed!"

    # Test case 2 - next-confirmed merged
    test_case_2_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True)],
        IS_GENERAL_REBASELINE: [(30, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 4)],
        IS_PIRA_REBASELINE: [(30, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 4)],
        IS_EVENT: [(30, True)],
        IS_ACCRUAL_EVENT: [(30, True)],
        EVENT_TYPE: [(30, LABEL_PIRA)],
        EVENT_SCORE: [(30, 4)],
        EVENT_REFERENCE_SCORE: [(30, 1.0)],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (65, 1),
            (70, 1),
        ],
        ACCRUAL_EVENT_ID: [
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
        targets_dict=test_case_2_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 2 failed!"

    # Tolerance, case 3
    test_dataframe_case_3 = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [
                1,
                1,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                3.5,
                4.0,
                4.5,
                4.5,
                5.0,
                5.5,
            ],
        }
    )
    test_case_3_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True), (110, True)],
        IS_GENERAL_REBASELINE: [(30, True), (80, True), (110, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (80, 4.5), (110, 5.5)],
        IS_PIRA_REBASELINE: [(30, True), (80, True), (110, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (80, 4.5), (110, 5.5)],
        IS_EVENT: [(30, True), (80, True), (110, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True), (110, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA), (110, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.5), (80, 4.5), (110, 5.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.5), (110, 4.5)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2), (110, 3)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2), (110, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_3,
        targets_dict=test_case_3_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
            "opt_baseline_type": "fixed",
        },
    ), "Test 3 failed!"

    # Tolerance, case 4
    test_case_4_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (110, True)],
        IS_GENERAL_REBASELINE: [(30, True), (110, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 4.5), (110, 5.5)],
        IS_PIRA_REBASELINE: [(30, True), (110, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 4.5), (110, 5.5)],
        IS_EVENT: [(30, True), (110, True)],
        IS_ACCRUAL_EVENT: [(30, True), (110, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (110, LABEL_PIRA)],
        EVENT_SCORE: [(30, 4.5), (110, 5.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (110, 4.5)],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (65, 1),
            (70, 1),
            (80, 1),
            (110, 2),
        ],
        ACCRUAL_EVENT_ID: [
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
        targets_dict=test_case_4_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 4 failed!"

    # Tolerance, case 5
    test_case_5_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True)],
        IS_GENERAL_REBASELINE: [(30, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 5.5)],
        IS_PIRA_REBASELINE: [(30, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 5.5)],
        IS_EVENT: [(30, True)],
        IS_ACCRUAL_EVENT: [(30, True)],
        EVENT_TYPE: [(30, LABEL_PIRA)],
        EVENT_SCORE: [(30, 5.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0)],
        EVENT_ID: [
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
        ACCRUAL_EVENT_ID: [
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
        targets_dict=test_case_5_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 5 failed!"

    # Test case 6 - dip, unconfirmed
    test_dataframe_case_6 = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [
                1,
                1,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                3.0,
                4.0,
                4.5,
                4.5,
                5.0,
                5.5,
            ],
        }
    )
    test_case_6_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True), (110, True)],
        IS_GENERAL_REBASELINE: [(30, True), (80, True), (110, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (80, 4.5), (110, 5.5)],
        IS_PIRA_REBASELINE: [(30, True), (80, True), (110, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (80, 4.5), (110, 5.5)],
        IS_EVENT: [(30, True), (80, True), (110, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True), (110, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA), (110, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.5), (80, 4.5), (110, 5.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.5), (110, 4.5)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2), (110, 3)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2), (110, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_6,
        targets_dict=test_case_6_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
            "opt_baseline_type": "fixed",
        },
    ), "Test 6 failed!"

    # Test case 7 - dip, unconfirmed
    test_case_7_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True), (110, True)],
        IS_GENERAL_REBASELINE: [(30, True), (80, True), (110, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (80, 4.5), (110, 5.5)],
        IS_PIRA_REBASELINE: [(30, True), (80, True), (110, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (80, 4.5), (110, 5.5)],
        IS_EVENT: [(30, True), (80, True), (110, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True), (110, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA), (110, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.5), (80, 4.5), (110, 5.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.5), (110, 4.5)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2), (110, 3)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2), (110, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_6,
        targets_dict=test_case_7_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 7 failed!"

    # Test case 8 - dip, unconfirmed
    test_case_8_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
        IS_GENERAL_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (80, 5.5)],
        IS_PIRA_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (80, 5.5)],
        IS_EVENT: [(30, True), (80, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.5), (80, 5.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.5)],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (80, 2),
            (90, 2),
            (100, 2),
            (110, 2),
        ],
        ACCRUAL_EVENT_ID: [
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
        targets_dict=test_case_8_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 8 failed!"

    # Test case 9 - dip, next-confirmed
    test_dataframe_case_9 = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [
                1,
                1,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                3.0,
                4.0,
                4.5,
                4.5,
                5.0,
                5.5,
            ],
        }
    )
    test_case_9_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (70, True)],
        IS_GENERAL_REBASELINE: [(30, True), (70, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3), (70, 4.5)],
        IS_PIRA_REBASELINE: [(30, True), (70, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3), (70, 4.5)],
        IS_EVENT: [(30, True), (70, True)],
        IS_ACCRUAL_EVENT: [(30, True), (70, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (70, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3), (70, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (70, 3.0)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (70, 2), (80, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (70, 2), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_9,
        targets_dict=test_case_9_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 9 failed!"

    # Test case 10 - dip, next-confirmed
    test_case_10_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (70, True)],
        IS_GENERAL_REBASELINE: [(30, True), (70, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.0), (70, 4.5)],
        IS_PIRA_REBASELINE: [(30, True), (70, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.0), (70, 4.5)],
        IS_EVENT: [(30, True), (70, True)],
        IS_ACCRUAL_EVENT: [(30, True), (70, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (70, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.0), (70, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (70, 3.0)],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (70, 2),
            (80, 2),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (70, 2),
            (80, 2),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_9,
        targets_dict=test_case_10_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 10 failed!"

    # Test case 11 - dip, next-confirmed
    test_case_11_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (70, True)],
        IS_GENERAL_REBASELINE: [(30, True), (70, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.0), (70, 5.0)],
        IS_PIRA_REBASELINE: [(30, True), (70, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.0), (70, 5.0)],
        IS_EVENT: [(30, True), (70, True)],
        IS_ACCRUAL_EVENT: [(30, True), (70, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (70, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.0), (70, 5.0)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (70, 3.0)],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (70, 2),
            (80, 2),
            (90, 2),
            (100, 2),
        ],
        ACCRUAL_EVENT_ID: [
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
        targets_dict=test_case_11_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 11 failed!"

    # Test case 12 - dip, 20 units confirmed
    test_dataframe_case_12 = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [
                1,
                1,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                3.0,
                4.0,
                4.5,
                4.5,
                5.0,
                5.5,
            ],
        }
    )
    test_case_12_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
        IS_GENERAL_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (80, 4.5)],
        IS_PIRA_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (80, 4.5)],
        IS_EVENT: [(30, True), (80, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.5), (80, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.5)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_12,
        targets_dict=test_case_12_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
            "opt_baseline_type": "fixed",
        },
    ), "Test 12 failed!"

    # Test case 13 - dip, 20 units confirmed
    test_case_13_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
        IS_GENERAL_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (80, 4.5)],
        IS_PIRA_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (80, 4.5)],
        IS_EVENT: [(30, True), (80, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.5), (80, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.5)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_12,
        targets_dict=test_case_13_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
            "opt_baseline_type": "fixed",
        },
    ), "Test 13 failed!"

    # Test case 14 - dip, 20 units confirmed
    test_case_14_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
        IS_GENERAL_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (80, 4.5)],
        IS_PIRA_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (80, 4.5)],
        IS_EVENT: [(30, True), (80, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.5), (80, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.5)],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (80, 2),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (80, 2),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_12,
        targets_dict=test_case_14_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
            "opt_baseline_type": "fixed",
        },
    ), "Test 14 failed!"

    # Test case 15 - dip with stagnation, next-confirmed
    test_dataframe_case_15 = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [
                1,
                1,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                3.0,
                3.0,
                4.5,
                4.5,
                5.0,
                5.5,
            ],
        }
    )
    test_case_15_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
        IS_GENERAL_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.0), (80, 4.5)],
        IS_PIRA_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.0), (80, 4.5)],
        IS_EVENT: [(30, True), (80, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.0), (80, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.0)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (80, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_15,
        targets_dict=test_case_15_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 15 failed!"

    # Test case 16 - dip with stagnation, next-confirmed
    test_case_16_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
        IS_GENERAL_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.0), (80, 4.5)],
        IS_PIRA_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.0), (80, 4.5)],
        IS_EVENT: [(30, True), (80, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.0), (80, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.0)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (80, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_15,
        targets_dict=test_case_16_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 16 failed!"

    # Test case 17 - dip with stagnation, next-confirmed
    test_case_17_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
        IS_GENERAL_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.0), (80, 5.0)],
        IS_PIRA_REBASELINE: [(30, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.0), (80, 5.0)],
        IS_EVENT: [(30, True), (80, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.0), (80, 5.0)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.0)],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (80, 2),
            (90, 2),
            (100, 2),
        ],
        ACCRUAL_EVENT_ID: [
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
        targets_dict=test_case_17_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 17 failed!"

    # Test case 18 - dip with stagnation, 20 units last confirmed
    test_dataframe_case_18 = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [
                1,
                1,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                3.0,
                3.0,
                4.5,
                4.5,
                5.0,
                5.5,
            ],
        }
    )
    test_case_18_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
        IS_GENERAL_REBASELINE: [(30, True), (65, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (65, 3.0), (80, 4.5)],
        IS_PIRA_REBASELINE: [(30, True), (65, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (65, 3.0), (80, 4.5)],
        IS_EVENT: [(30, True), (80, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.5), (80, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.0)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_18,
        targets_dict=test_case_18_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
        },
    ), "Test 18 failed!"

    # Test case 19 - dip with stagnation, 20 units last confirmed
    test_case_19_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
        IS_GENERAL_REBASELINE: [(30, True), (65, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (65, 3.0), (80, 4.5)],
        IS_PIRA_REBASELINE: [(30, True), (65, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (65, 3.0), (80, 4.5)],
        IS_EVENT: [(30, True), (80, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.5), (80, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.0)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_18,
        targets_dict=test_case_19_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
        },
    ), "Test 19 failed!"

    # Test case 20 - dip with stagnation, 20 units last confirmed
    test_case_20_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
        IS_GENERAL_REBASELINE: [(30, True), (65, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (65, 3.0), (80, 4.5)],
        IS_PIRA_REBASELINE: [(30, True), (65, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (65, 3.0), (80, 4.5)],
        IS_EVENT: [(30, True), (80, True)],
        IS_ACCRUAL_EVENT: [(30, True), (80, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.5), (80, 4.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (80, 3.0)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_18,
        targets_dict=test_case_20_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
            "opt_confirmation_included_values": "last",
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
        },
    ), "Test 20 failed!"

    # Inverted mode
    test_dataframe_inv = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 50, 60, 65, 70],
            EDSS_SCORE: [5.0, 5.0, 4.5, 4.0, 3.0, 2.5, 2.0, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (50, True), (65, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True), (65, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 4.0),
                (50, 3.0),
                (65, 2.0),
            ],
            # IS_PIRA_REBASELINE: [(30, True), (50, True), (65, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 4.0),
            #    (50, 3.0),
            #    (65, 2.0),
            # ],
            IS_EVENT: [(30, True), (50, True), (65, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (50, True), (65, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (50, LABEL_IMPROVEMENT),
                (65, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 4.0),
                (50, 3.0),
                (65, 2.0),
            ],
            EVENT_REFERENCE_SCORE: [
                (30, 5.0),
                (50, 4.0),
                (65, 3.0),
            ],
            EVENT_ID: [(30, 1), (50, 2), (65, 3)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (50, 2), (65, 3)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 21 failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (50, True)],
            IS_GENERAL_REBASELINE: [(30, True), (50, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 4.0),
                (50, 1.5),
            ],
            # IS_PIRA_REBASELINE: [(30, True), (50, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 4.0),
            #    (50, 1.5),
            # ],
            IS_EVENT: [(30, True), (50, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (50, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (50, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 4.0),
                (50, 1.5),
            ],
            EVENT_REFERENCE_SCORE: [
                (30, 5.0),
                (50, 4.0),
            ],
            EVENT_ID: [(30, 1), (50, 2), (60, 2), (65, 2), (70, 2)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (50, 2), (60, 2), (65, 2), (70, 2)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_merge_distance": 10,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 22 failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 1.5),
            ],
            # IS_PIRA_REBASELINE: [(30, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 1.5),
            # ],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 1.5),
            ],
            EVENT_REFERENCE_SCORE: [
                (30, 5.0),
            ],
            EVENT_ID: [(30, 1), (50, 1), (60, 1), (65, 1), (70, 1)],
            IMPROVEMENT_EVENT_ID: [(30, 1), (50, 1), (60, 1), (65, 1), (70, 1)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "merge_continuous_events": True,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
            "opt_baseline_type": "fixed",
        },
    ), "Test 23 failed!"
    test_dataframe_inv_rep = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [
                5.5,
                5.5,
                5.0,
                4.5,
                4.0,
                3.5,
                3.0,
                3.0,
                2.5,
                2.0,
                2.0,
                1.5,
                1,
            ],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_rep,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (80, True), (110, True)],
            IS_GENERAL_REBASELINE: [(30, True), (80, True), (110, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 3.0),
                (80, 2.0),
                (110, 1.0),
            ],
            # IS_PIRA_REBASELINE: [(30, True), (80, True), (110, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 3.0),
            #    (80, 2.0),
            #    (110, 1.0),
            # ],
            IS_EVENT: [(30, True), (80, True), (110, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (80, True), (110, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (80, LABEL_IMPROVEMENT),
                (110, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 3.0),
                (80, 2.0),
                (110, 1.0),
            ],
            EVENT_REFERENCE_SCORE: [
                (30, 5.5),
                (80, 3.0),
                (110, 2.0),
            ],
            EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (80, 2), (110, 3)],
            IMPROVEMENT_EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (80, 2),
                (110, 3),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 4,
            "opt_baseline_type": "fixed",
        },
    ), "Test 24 failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_rep,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (110, True)],
            IS_GENERAL_REBASELINE: [(30, True), (110, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 2.0),
                (110, 1.0),
            ],
            # IS_PIRA_REBASELINE: [(30, True), (110, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 2.0),
            #    (110, 1.0),
            # ],
            IS_EVENT: [(30, True), (110, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (110, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (110, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 2.0),
                (110, 1.0),
            ],
            EVENT_REFERENCE_SCORE: [
                (30, 5.5),
                (110, 2.0),
            ],
            EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (65, 1),
                (70, 1),
                (80, 1),
                (110, 2),
            ],
            IMPROVEMENT_EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (65, 1),
                (70, 1),
                (80, 1),
                (110, 2),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 25 failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_rep,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 1.0),
            ],
            # IS_PIRA_REBASELINE: [(30, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 1.0),
            # ],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 1.0),
            ],
            EVENT_REFERENCE_SCORE: [
                (30, 5.5),
            ],
            EVENT_ID: [
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
            IMPROVEMENT_EVENT_ID: [
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
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 26 failed!"
    test_dataframe_inv_stag = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [
                5.5,
                5.5,
                5.0,
                4.5,
                4.0,
                3.5,
                3.0,
                3.0,
                3.0,
                2.0,
                2.0,
                1.5,
                1,
            ],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_stag,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (80, True), (110, True)],
            IS_GENERAL_REBASELINE: [(30, True), (80, True), (110, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 3.0),
                (80, 2.0),
                (110, 1.0),
            ],
            # IS_PIRA_REBASELINE: [(30, True), (80, True), (110, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 3.0),
            #    (80, 2.0),
            #    (110, 1.0),
            # ],
            IS_EVENT: [(30, True), (80, True), (110, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (80, True), (110, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (80, LABEL_IMPROVEMENT),
                (110, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 3.0),
                (80, 2.0),
                (110, 1.0),
            ],
            EVENT_REFERENCE_SCORE: [
                (30, 5.5),
                (80, 3.0),
                (110, 2.0),
            ],
            EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (80, 2),
                (110, 3),
            ],
            IMPROVEMENT_EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (80, 2),
                (110, 3),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_baseline_type": "fixed",
        },
    ), "Test 27 failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_stag,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 1.0),
            ],
            # IS_PIRA_REBASELINE: [(30, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 1.0),
            # ],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 1.0),
            ],
            EVENT_REFERENCE_SCORE: [
                (30, 5.5),
            ],
            EVENT_ID: [
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
            IMPROVEMENT_EVENT_ID: [
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
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 28 failed!"
    test_dataframe_inv_stag_end = pd.DataFrame(
        {
            TIMESTAMP: [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                65,
                70,
            ],
            EDSS_SCORE: [4.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.5, 1.5],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_stag_end,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (30, 1.5),
            ],
            # IS_PIRA_REBASELINE: [(30, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            #    (30, 1.5),
            # ],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [
                (30, 1.5),
            ],
            EVENT_REFERENCE_SCORE: [
                (30, 4.0),
            ],
            EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
            ],
            IMPROVEMENT_EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_baseline_type": "fixed",
        },
    ), "Test 29 failed!"
    test_dataframe_inv_impr = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [
                5.5,
                5.5,
                5.0,
                4.5,
                4.0,
                3.5,
                3.0,
                3.5,
                2.5,
                2.0,
                2.0,
                1.5,
                1,
            ],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_impr,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
            IS_GENERAL_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.0), (80, 1.0)],
            # IS_PIRA_REBASELINE: [(30, True), (80, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.0), (80, 1.0)],
            IS_EVENT: [(30, True), (80, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (80, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (80, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [(30, 3.0), (80, 1.0)],
            EVENT_REFERENCE_SCORE: [(30, 5.5), (80, 3.0)],
            EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (80, 2),
                (90, 2),
                (100, 2),
                (110, 2),
            ],
            IMPROVEMENT_EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (80, 2),
                (90, 2),
                (100, 2),
                (110, 2),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 30 failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_impr,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 1.5)],
            # IS_PIRA_REBASELINE: [(30, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 1.5)],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [(30, 1.5)],
            EVENT_REFERENCE_SCORE: [(30, 5.5)],
            EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (65, 1),
                (70, 1),
                (80, 1),
                (90, 1),
                (100, 1),
            ],
            IMPROVEMENT_EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (65, 1),
                (70, 1),
                (80, 1),
                (90, 1),
                (100, 1),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 31 failed!"
    test_dataframe_inv_impr.at[7, EDSS_SCORE] = 4.0
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_impr,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (70, True)],
            IS_GENERAL_REBASELINE: [(30, True), (70, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (70, 1.5)],
            # IS_PIRA_REBASELINE: [(30, True), (70, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (70, 1.5)],
            IS_EVENT: [(30, True), (70, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (70, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (70, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [(30, 3.5), (70, 1.5)],
            EVENT_REFERENCE_SCORE: [(30, 5.5), (70, 3.5)],
            EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (70, 2),
                (80, 2),
                (90, 2),
                (100, 2),
            ],
            IMPROVEMENT_EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (70, 2),
                (80, 2),
                (90, 2),
                (100, 2),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 32 failed!"
    test_dataframe_inv_conf = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [
                5.5,
                5.5,
                5.0,
                4.5,
                4.0,
                3.5,
                3.0,
                3.5,
                3.5,
                2.0,
                2.0,
                1.5,
                1,
            ],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_conf,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 1.5)],
            # IS_PIRA_REBASELINE: [(30, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 1.5)],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [(30, 1.5)],
            EVENT_REFERENCE_SCORE: [(30, 5.5)],
            EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (65, 1),
                (70, 1),
                (80, 1),
                (90, 1),
                (100, 1),
            ],
            IMPROVEMENT_EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (65, 1),
                (70, 1),
                (80, 1),
                (90, 1),
                (100, 1),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 33 failed!"
    test_dataframe_inv_conf.at[8, EDSS_SCORE] = 4.0
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_conf,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
            IS_GENERAL_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5), (80, 1.5)],
            # IS_PIRA_REBASELINE: [(30, True), (80, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.5), (80, 1.5)],
            IS_EVENT: [(30, True), (80, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (80, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (80, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [(30, 3.5), (80, 1.5)],
            EVENT_REFERENCE_SCORE: [(30, 5.5), (80, 3.5)],
            EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (80, 2),
                (90, 2),
                (100, 2),
            ],
            IMPROVEMENT_EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (80, 2),
                (90, 2),
                (100, 2),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 20,
            "opt_baseline_type": "fixed",
        },
    ), "Test 34 failed!"
    test_dataframe_inv_conf_lst = pd.DataFrame(
        {
            TIMESTAMP: [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                65,
                70,
                75,
                80,
                90,
                100,
                110,
            ],
            EDSS_SCORE: [
                1,
                1,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                3.0,
                2.5,
                3.0,
                4.5,
                4.5,
                5.0,
                5.5,
            ],
        }
    )
    test_dataframe_inv_conf_lst[EDSS_SCORE] = (
        6.5 - test_dataframe_inv_conf_lst[EDSS_SCORE]
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_conf_lst,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True)],
            IS_GENERAL_REBASELINE: [(30, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
            # IS_PIRA_REBASELINE: [(30, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0)],
            IS_EVENT: [(30, True)],
            IS_IMPROVEMENT_EVENT: [(30, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [(30, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 5.5)],
            EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (65, 1),
                (70, 1),
                (75, 1),
                (80, 1),
            ],
            IMPROVEMENT_EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (65, 1),
                (70, 1),
                (75, 1),
                (80, 1),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 30,
            "opt_baseline_type": "fixed",
        },
    ), "Test 35 failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_inv_conf_lst,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(30, True), (80, True)],
            IS_GENERAL_REBASELINE: [(30, True), (80, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.0), (80, 2.0)],
            # IS_PIRA_REBASELINE: [(30, True), (80, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3.0), (80, 2.0)],
            IS_EVENT: [(30, True), (80, True)],
            IS_IMPROVEMENT_EVENT: [(30, True), (80, True)],
            EVENT_TYPE: [
                (30, LABEL_IMPROVEMENT),
                (80, LABEL_IMPROVEMENT),
            ],
            EVENT_SCORE: [(30, 3.0), (80, 2.0)],
            EVENT_REFERENCE_SCORE: [(30, 5.5), (80, 3.0)],
            EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (80, 2),
            ],
            IMPROVEMENT_EVENT_ID: [
                (30, 1),
                (40, 1),
                (50, 1),
                (60, 1),
                (80, 2),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "last",
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 30,
            "opt_baseline_type": "fixed",
        },
    ), "Test 36 failed!"

    # TODO: Symmetric mode? -> make sure increase/decrease
    # are not merged.
    test_dataframe_merge_symm = pd.DataFrame(
        {
            TIMESTAMP: [10 * i for i in range(11)],
            EDSS_SCORE: [7.5, 7.0, 6.5, 6.0, 6.5, 7.0, 6.0, 5.5, 4.5, 5.5, 6.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_merge_symm[
            test_dataframe_merge_symm[TIMESTAMP] != 35
        ],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True), (40, True), (60, True), (90, True)],
            IS_GENERAL_REBASELINE: [(10, True), (40, True), (60, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (10, 6.0),
                (40, 7.0),
                (60, 4.5),
                (90, 6.0),
            ],
            IS_PIRA_REBASELINE: [(10, True), (40, True), (60, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (10, 6.0),
                (40, 7.0),
                (60, 4.5),
                (90, 6.0),
            ],
            IS_EVENT: [(10, True), (40, True), (60, True), (90, True)],
            IS_ACCRUAL_EVENT: [(40, True), (90, True)],
            IS_IMPROVEMENT_EVENT: [(10, True), (60, True)],
            EVENT_TYPE: [
                (10, LABEL_IMPROVEMENT),
                (40, LABEL_PIRA),
                (60, LABEL_IMPROVEMENT),
                (90, LABEL_PIRA),
            ],
            EVENT_SCORE: [(10, 6.0), (40, 7.0), (60, 4.5), (90, 6.0)],
            EVENT_REFERENCE_SCORE: [(10, 7.5), (40, 6.0), (60, 7.0), (90, 4.5)],
            EVENT_ID: [
                (10, 1),
                (20, 1),
                (30, 1),
                (40, 2),
                (50, 2),
                (60, 3),
                (70, 3),
                (80, 3),
                (90, 4),
                (100, 4),
            ],
            IMPROVEMENT_EVENT_ID: [
                (10, 1),
                (20, 1),
                (30, 1),
                (60, 2),
                (70, 2),
                (80, 2),
            ],
            ACCRUAL_EVENT_ID: [
                (40, 1),
                (50, 1),
                (100, 2),
                (90, 2),
            ],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 30,
            "opt_baseline_type": "fixed",
        },
    ), "Test 37 failed!"
    test_dataframe_merge_symm_stag = pd.DataFrame(
        {
            TIMESTAMP: [10 * i for i in range(4)]
            + [35]
            + [10 * i for i in range(4, 11)],
            EDSS_SCORE: [7.5, 7.0, 6.5, 6.0, 6.0, 6.5, 7.0, 6.0, 5.5, 4.5, 5.5, 6.0],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_merge_symm_stag,
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(10, True), (40, True), (60, True), (90, True)],
            IS_GENERAL_REBASELINE: [(10, True), (40, True), (60, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
                (10, 6.0),
                (40, 7.0),
                (60, 4.5),
                (90, 6.0),
            ],
            IS_PIRA_REBASELINE: [(10, True), (40, True), (60, True), (90, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (10, 6.0),
                (40, 7.0),
                (60, 4.5),
                (90, 6.0),
            ],
            IS_EVENT: [(10, True), (40, True), (60, True), (90, True)],
            IS_ACCRUAL_EVENT: [(40, True), (90, True)],
            IS_IMPROVEMENT_EVENT: [(10, True), (60, True)],
            EVENT_TYPE: [
                (10, LABEL_IMPROVEMENT),
                (40, LABEL_PIRA),
                (60, LABEL_IMPROVEMENT),
                (90, LABEL_PIRA),
            ],
            EVENT_SCORE: [(10, 6.0), (40, 7.0), (60, 4.5), (90, 6.0)],
            EVENT_REFERENCE_SCORE: [(10, 7.5), (40, 6.0), (60, 7.0), (90, 4.5)],
            EVENT_ID: [
                (10, 1),
                (20, 1),
                (30, 1),
                (40, 2),
                (50, 2),
                (60, 3),
                (70, 3),
                (80, 3),
                (90, 4),
                (100, 4),
            ],
            IMPROVEMENT_EVENT_ID: [
                (10, 1),
                (20, 1),
                (30, 1),
                (60, 2),
                (70, 2),
                (80, 2),
            ],
            ACCRUAL_EVENT_ID: [
                (40, 1),
                (50, 1),
                (100, 2),
                (90, 2),
            ],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 30,
            "opt_baseline_type": "fixed",
        },
    ), "Test 38 failed!"


# ----------------------
# Part 3 - with relapses
# ----------------------


def test_add_relapses_to_follow_up():
    test_dataframe = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60, 70],
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.0, 1.5, 1.5, 2.5],
        }
    )

    # Test case 1 - no relapses, must yield empty dataframe
    target_case_1 = test_dataframe.copy()
    target_case_1[DAYS_SINCE_PREVIOUS_RELAPSE] = np.nan
    target_case_1[DAYS_TO_NEXT_RELAPSE] = np.nan
    assert (
        edssannotation.EDSSAnnotation()
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
                    DAYS_SINCE_PREVIOUS_RELAPSE: [
                        np.nan,
                        np.nan,
                        0,
                        10,
                        20,
                        30,
                        40,
                        50,
                    ],
                    DAYS_TO_NEXT_RELAPSE: [
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
        edssannotation.EDSSAnnotation()
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
                    DAYS_SINCE_PREVIOUS_RELAPSE: [
                        np.nan,
                        np.nan,
                        np.nan,
                        5,
                        15,
                        25,
                        35,
                        45,
                    ],
                    DAYS_TO_NEXT_RELAPSE: [
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
        edssannotation.EDSSAnnotation()
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
                    DAYS_SINCE_PREVIOUS_RELAPSE: [
                        np.nan,
                        2,
                        12,
                        22,
                        32,
                        42,
                        52,
                        62,
                    ],
                    DAYS_TO_NEXT_RELAPSE: [
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
        edssannotation.EDSSAnnotation()
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
                    DAYS_SINCE_PREVIOUS_RELAPSE: [
                        5,
                        15,
                        25,
                        35,
                        45,
                        55,
                        65,
                        75,
                    ],
                    DAYS_TO_NEXT_RELAPSE: [
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
        edssannotation.EDSSAnnotation()
        ._add_relapses_to_follow_up(
            follow_up_df=test_dataframe,
            relapse_timestamps=[-5, 80],
        )
        .equals(target_case_5)
    ), "Test 5 'Relapses at -5 and 80' failed!"


def test_get_post_relapse_rebaseline_timestamps():
    test_relapses_cases_1_2_3_4 = [6, 30]
    test_dataframe_cases_1_2_3_4 = pd.DataFrame({TIMESTAMP: [0, 8, 36, 48, 60]})
    # Test case 1 - assessments well-separated
    test_case_1_target = [8, 36]
    assert Counter(
        edssannotation.EDSSAnnotation(
            opt_raw_before_relapse_max_time=2,
            opt_raw_after_relapse_max_time=1,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_1_2_3_4,
            relapse_timestamps=test_relapses_cases_1_2_3_4,
        )
    ) == Counter(test_case_1_target), "Test 1 'Well-separated re-baselining' failed!"

    # Test case 2 - rebaseline of first relapse after second
    # relapse, within buffer
    test_case_2_target = [36, 48]
    assert Counter(
        edssannotation.EDSSAnnotation(
            opt_raw_before_relapse_max_time=4,
            opt_raw_after_relapse_max_time=12,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_1_2_3_4,
            relapse_timestamps=test_relapses_cases_1_2_3_4,
        )
    ) == Counter(test_case_2_target), (
        "Test 2 'Rebaseline of first relapse after second relapse, within buffer' failed!"
    )

    # Test case 3 - rebaseline of first relapse after second relapse,
    # after second buffer (same for both)
    test_case_3_target = [36]
    assert Counter(
        edssannotation.EDSSAnnotation(
            opt_raw_before_relapse_max_time=4,
            opt_raw_after_relapse_max_time=4,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_1_2_3_4,
            relapse_timestamps=test_relapses_cases_1_2_3_4,
        )
    ) == Counter(test_case_3_target), (
        "Test 3 'Rebaseline of first relapse after second relapse, after second buffer' failed!"
    )

    # Test case 4 - overlapping RAW windows
    test_case_4_target = [60]
    assert Counter(
        edssannotation.EDSSAnnotation(
            opt_raw_before_relapse_max_time=4,
            opt_raw_after_relapse_max_time=20,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_1_2_3_4,
            relapse_timestamps=test_relapses_cases_1_2_3_4,
        )
    ) == Counter(test_case_4_target), "Test 4 'Overlapping RAW windows failed!"

    # Test case 5 - rebaseline of first relapse before second, but within RAW window
    test_relapses_cases_5 = [6, 30]
    test_dataframe_cases_5 = pd.DataFrame({TIMESTAMP: [0, 8, 28, 48, 60]})
    test_case_5_target = [28, 48]
    assert Counter(
        edssannotation.EDSSAnnotation(
            opt_raw_before_relapse_max_time=4,
            opt_raw_after_relapse_max_time=4,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_5,
            relapse_timestamps=test_relapses_cases_5,
        )
    ) == Counter(test_case_5_target), (
        "Test 5 'Rebaseline within buffer of next' failed!"
    )

    # Test case 6 - multiple non-overlapping and overlapping relapses
    test_relapses_cases_6 = [15, 25, 40, 48]
    test_dataframe_cases_6 = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 50, 60, 70],
        }
    )
    test_case_6_target = [50, 60]
    assert Counter(
        edssannotation.EDSSAnnotation(
            opt_raw_before_relapse_max_time=2,
            opt_raw_after_relapse_max_time=7,
        )._get_post_relapse_rebaseline_timestamps(
            follow_up_df=test_dataframe_cases_6,
            relapse_timestamps=test_relapses_cases_6,
        )
    ) == Counter(test_case_6_target), (
        "Test 6 'Rebaseline with multiple overlapping and non-overlapping' failed!"
    )


# Event types
def check_event_types():
    # RAW
    raw_df = pd.DataFrame(
        {
            TIMESTAMP: [0, 90, 180, 270, 360, 450],
            EDSS_SCORE: [1, 1, 2.0, 2, 2, 2],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=raw_df,
        relapse_timestamps=[160],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (180, True),
            ],
            # IS_POST_RELAPSE_REBASELINE: [(90, True)],
            IS_GENERAL_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(180, 2.0)],
            IS_PIRA_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(180, 2.0)],
            IS_EVENT: [(180, True)],
            IS_ACCRUAL_EVENT: [(180, True)],
            EVENT_TYPE: [(180, LABEL_RAW)],
            EVENT_SCORE: [(180, 2.0)],
            EVENT_REFERENCE_SCORE: [(180, 1.0)],
            EVENT_ID: [(180, 1.0)],
            ACCRUAL_EVENT_ID: [(180, 1.0)],
            DAYS_TO_NEXT_RELAPSE: [(0, 160), (90, 70)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (180, 20),
                (270, 110),
                (360, 200),
                (450, 290),
            ],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 1 RAW in symmetric mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=raw_df,
        relapse_timestamps=[160],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (180, True),
            ],
            # IS_POST_RELAPSE_REBASELINE: [(90, True)],
            IS_GENERAL_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(180, 2.0)],
            IS_PIRA_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(180, 2.0)],
            IS_EVENT: [(180, True)],
            IS_ACCRUAL_EVENT: [(180, True)],
            EVENT_TYPE: [(180, LABEL_RAW)],
            EVENT_SCORE: [(180, 2.0)],
            EVENT_REFERENCE_SCORE: [(180, 1.0)],
            EVENT_ID: [(180, 1.0)],
            ACCRUAL_EVENT_ID: [(180, 1.0)],
            DAYS_TO_NEXT_RELAPSE: [(0, 160), (90, 70)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (180, 20),
                (270, 110),
                (360, 200),
                (450, 290),
            ],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 2 RAW in accrual mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=raw_df,
        relapse_timestamps=[160],
        targets_dict={
            DAYS_TO_NEXT_RELAPSE: [(0, 160), (90, 70)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (180, 20),
                (270, 110),
                (360, 200),
                (450, 290),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 3 RAW in improvement mode failed!"
    # PIRA
    pira_df = pd.DataFrame(
        {
            TIMESTAMP: [0, 90, 180, 270, 360, 450],
            EDSS_SCORE: [1, 1.5, 2.0, 2, 2, 2],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pira_df,
        relapse_timestamps=[],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (180, True),
            ],
            IS_GENERAL_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(180, 2.0)],
            IS_PIRA_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(180, 2.0)],
            IS_EVENT: [(180, True)],
            IS_ACCRUAL_EVENT: [(180, True)],
            EVENT_TYPE: [(180, LABEL_PIRA)],
            EVENT_SCORE: [(180, 2.0)],
            EVENT_REFERENCE_SCORE: [(180, 1.0)],
            EVENT_ID: [(180, 1.0)],
            ACCRUAL_EVENT_ID: [(180, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 4 PIRA in symmetric mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pira_df,
        relapse_timestamps=[],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (180, True),
            ],
            IS_GENERAL_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(180, 2.0)],
            IS_PIRA_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(180, 2.0)],
            IS_EVENT: [(180, True)],
            IS_ACCRUAL_EVENT: [(180, True)],
            EVENT_TYPE: [(180, LABEL_PIRA)],
            EVENT_SCORE: [(180, 2.0)],
            EVENT_REFERENCE_SCORE: [(180, 1.0)],
            EVENT_ID: [(180, 1.0)],
            ACCRUAL_EVENT_ID: [(180, 1.0)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 5 PIRA in accrual mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pira_df,
        relapse_timestamps=[],
        targets_dict={},
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 6 PIRA in improvement mode failed!"
    # Improvement
    improvement_df = pd.DataFrame(
        {
            TIMESTAMP: [0, 90, 180, 270, 360, 450],
            EDSS_SCORE: [2, 1.5, 1, 1, 1, 1],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=improvement_df,
        relapse_timestamps=[],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (180, True),
            ],
            IS_GENERAL_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(180, 1.0)],
            IS_PIRA_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(180, 1.0)],
            IS_EVENT: [(180, True)],
            IS_IMPROVEMENT_EVENT: [(180, True)],
            EVENT_TYPE: [(180, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(180, 1.0)],
            EVENT_REFERENCE_SCORE: [(180, 2.0)],
            EVENT_ID: [(180, 1.0)],
            IMPROVEMENT_EVENT_ID: [(180, 1.0)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 7 improvement in symmetric mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=improvement_df,
        relapse_timestamps=[],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (180, True),
            ],
            IS_GENERAL_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(180, 1.0)],
            # IS_PIRA_REBASELINE: [(180, True)],
            # EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(180, 1.0)],
            IS_EVENT: [(180, True)],
            IS_IMPROVEMENT_EVENT: [(180, True)],
            EVENT_TYPE: [(180, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(180, 1.0)],
            EVENT_REFERENCE_SCORE: [(180, 2.0)],
            EVENT_ID: [(180, 1.0)],
            IMPROVEMENT_EVENT_ID: [(180, 1.0)],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 8 improvement in improvement mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=improvement_df,
        relapse_timestamps=[],
        targets_dict={},
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 9 improvement in accrual mode failed!"
    # PIRA with relapse during confirmation
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pira_df,
        relapse_timestamps=[290],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (180, True),
            ],
            IS_GENERAL_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(180, 2.0)],
            IS_PIRA_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(180, 2.0)],
            IS_EVENT: [(180, True)],
            IS_ACCRUAL_EVENT: [(180, True)],
            EVENT_TYPE: [(180, LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW)],
            EVENT_SCORE: [(180, 2.0)],
            EVENT_REFERENCE_SCORE: [(180, 1.0)],
            EVENT_ID: [(180, 1.0)],
            ACCRUAL_EVENT_ID: [(180, 1.0)],
            DAYS_TO_NEXT_RELAPSE: [(0, 290), (90, 200), (180, 110), (270, 20)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (360, 70),
                (450, 160),
            ],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 10 PIRA WRC in symmetric mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pira_df,
        relapse_timestamps=[290],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (180, True),
            ],
            IS_GENERAL_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(180, 2.0)],
            IS_PIRA_REBASELINE: [(180, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(180, 2.0)],
            IS_EVENT: [(180, True)],
            IS_ACCRUAL_EVENT: [(180, True)],
            EVENT_TYPE: [(180, LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW)],
            EVENT_SCORE: [(180, 2.0)],
            EVENT_REFERENCE_SCORE: [(180, 1.0)],
            EVENT_ID: [(180, 1.0)],
            ACCRUAL_EVENT_ID: [(180, 1.0)],
            DAYS_TO_NEXT_RELAPSE: [(0, 290), (90, 200), (180, 110), (270, 20)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (360, 70),
                (450, 160),
            ],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 11 PIRA WRC in accrual mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=pira_df,
        relapse_timestamps=[290],
        targets_dict={
            DAYS_TO_NEXT_RELAPSE: [(0, 290), (90, 200), (180, 110), (270, 20)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (360, 70),
                (450, 160),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 12 PIRA WRC in improvement mode failed!"
    # Undefined as re-baselining
    undefined_df_1 = pd.DataFrame(
        {
            TIMESTAMP: [0, 90, 270, 360, 450],
            EDSS_SCORE: [
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
            ],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=undefined_df_1,
        relapse_timestamps=[160],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (270, True),
            ],
            IS_POST_RELAPSE_REBASELINE: [(270, True)],
            IS_GENERAL_REBASELINE: [(270, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(270, 2.0)],
            IS_PIRA_REBASELINE: [(270, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(270, 2.0)],
            IS_EVENT: [(270, True)],
            IS_ACCRUAL_EVENT: [(270, True)],
            EVENT_TYPE: [(270, LABEL_UNDEFINED)],
            EVENT_SCORE: [(270, 2.0)],
            EVENT_REFERENCE_SCORE: [(270, 1.0)],
            EVENT_ID: [(270, 1.0)],
            ACCRUAL_EVENT_ID: [(270, 1.0)],
            DAYS_TO_NEXT_RELAPSE: [(0, 160), (90, 70)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (270, 110),
                (360, 200),
                (450, 290),
            ],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 13 Undefined PRRB in symmetric mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=undefined_df_1,
        relapse_timestamps=[160],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (270, True),
            ],
            IS_POST_RELAPSE_REBASELINE: [(270, True)],
            IS_GENERAL_REBASELINE: [(270, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(270, 2.0)],
            IS_PIRA_REBASELINE: [(270, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(270, 2.0)],
            IS_EVENT: [(270, True)],
            IS_ACCRUAL_EVENT: [(270, True)],
            EVENT_TYPE: [(270, LABEL_UNDEFINED)],
            EVENT_SCORE: [(270, 2.0)],
            EVENT_REFERENCE_SCORE: [(270, 1.0)],
            EVENT_ID: [(270, 1.0)],
            ACCRUAL_EVENT_ID: [(270, 1.0)],
            DAYS_TO_NEXT_RELAPSE: [(0, 160), (90, 70)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (270, 110),
                (360, 200),
                (450, 290),
            ],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 14 Undefined PRRB in accrual mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=undefined_df_1,
        relapse_timestamps=[160],
        targets_dict={
            DAYS_TO_NEXT_RELAPSE: [(0, 160), (90, 70)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (270, 110),
                (360, 200),
                (450, 290),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 15 Undefined PRRB in improvement mode failed!"
    # Undefined as mixed
    undefined_df_2 = pd.DataFrame(
        {
            TIMESTAMP: [0, 90, 270, 360, 450],
            EDSS_SCORE: [
                1.0,
                1.0,
                1.5,
                2.0,
                2.0,
            ],
        }
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=undefined_df_2,
        relapse_timestamps=[160],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [(360, True)],
            IS_POST_RELAPSE_REBASELINE: [(270, True)],
            IS_GENERAL_REBASELINE: [(360, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(360, 2.0)],
            IS_PIRA_REBASELINE: [(270, True), (360, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(270, 1.5), (360, 2.0)],
            IS_EVENT: [(360, True)],
            IS_ACCRUAL_EVENT: [(360, True)],
            EVENT_TYPE: [(360, LABEL_UNDEFINED)],
            EVENT_SCORE: [(360, 2.0)],
            EVENT_REFERENCE_SCORE: [(360, 1.0)],
            EVENT_ID: [(360, 1.0)],
            ACCRUAL_EVENT_ID: [(360, 1.0)],
            DAYS_TO_NEXT_RELAPSE: [(0, 160), (90, 70)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (270, 110),
                (360, 200),
                (450, 290),
            ],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 16 Undefined mixed type in symmetric mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=undefined_df_2,
        relapse_timestamps=[160],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (360, True),
            ],
            IS_POST_RELAPSE_REBASELINE: [(270, True)],
            IS_GENERAL_REBASELINE: [(360, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(360, 2.0)],
            IS_PIRA_REBASELINE: [(270, True), (360, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(270, 1.5), (360, 2.0)],
            IS_EVENT: [(360, True)],
            IS_ACCRUAL_EVENT: [(360, True)],
            EVENT_TYPE: [(360, LABEL_UNDEFINED)],
            EVENT_SCORE: [(360, 2.0)],
            EVENT_REFERENCE_SCORE: [(360, 1.0)],
            EVENT_ID: [(360, 1.0)],
            ACCRUAL_EVENT_ID: [(360, 1.0)],
            DAYS_TO_NEXT_RELAPSE: [(0, 160), (90, 70)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (270, 110),
                (360, 200),
                (450, 290),
            ],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 17 Undefined mixed type in accrual mode failed!"
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=undefined_df_2,
        relapse_timestamps=[160],
        targets_dict={
            DAYS_TO_NEXT_RELAPSE: [(0, 160), (90, 70)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (270, 110),
                (360, 200),
                (450, 290),
            ],
        },
        args_dict={
            "annotation_mode": IMPROVEMENT_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 24 * 7,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 90,
        },
    ), "Test 18 Undefined mixed type in improvement mode failed!"


# Symmetric mode must not have improvement with respect to
# the PIRA baseline.
def check_no_improvement_with_respect_to_pira_baseline():
    # Symmetric mode
    test_dataframe = pd.DataFrame(
        [
            {TIMESTAMP: 0, EDSS_SCORE: 2.0},
            {TIMESTAMP: 60, EDSS_SCORE: 3.5},
            {TIMESTAMP: 90, EDSS_SCORE: 2.5},
            {TIMESTAMP: 150, EDSS_SCORE: 2.5},
            {TIMESTAMP: 220, EDSS_SCORE: 1.5},
            {TIMESTAMP: 250, EDSS_SCORE: 1.0},
            {TIMESTAMP: 310, EDSS_SCORE: 1.0},
        ]
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[40],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (250, True),
            ],
            IS_POST_RELAPSE_REBASELINE: [(90, True)],
            IS_GENERAL_REBASELINE: [(250, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(250, 1.0)],
            IS_PIRA_REBASELINE: [(90, True), (250, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(90, 2.5), (250, 1.0)],
            IS_EVENT: [(250, True)],
            IS_IMPROVEMENT_EVENT: [(250, True)],
            EVENT_TYPE: [(250, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(250, 1.0)],
            EVENT_REFERENCE_SCORE: [(250, 2.0)],
            EVENT_ID: [(250, 1.0)],
            IMPROVEMENT_EVENT_ID: [(250, 1.0)],
            DAYS_TO_NEXT_RELAPSE: [(0, 40)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (60, 20),
                (90, 50),
                (150, 110),
                (220, 180),
                (250, 210),
                (310, 270),
            ],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 30,
        },
    ), "Test 1 symmetric mode failed!"
    # Same, but improvement with roving reference; this
    # should then be an event at 220 as if it were
    # w.r.t. PIRA in symmetric mode.
    test_dataframe = pd.DataFrame(
        [
            {TIMESTAMP: 0, EDSS_SCORE: 2.0},
            {TIMESTAMP: 60, EDSS_SCORE: 3.5},
            {TIMESTAMP: 90, EDSS_SCORE: 2.5},
            {TIMESTAMP: 150, EDSS_SCORE: 2.5},
            {TIMESTAMP: 220, EDSS_SCORE: 1.5},
            {TIMESTAMP: 250, EDSS_SCORE: 1.0},
            {TIMESTAMP: 310, EDSS_SCORE: 1.0},
        ]
    )
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[40],
        targets_dict={
            IS_POST_EVENT_REBASELINE: [
                (250, True),
            ],
            IS_POST_RELAPSE_REBASELINE: [(90, True)],
            IS_GENERAL_REBASELINE: [(250, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(250, 1.0)],
            IS_PIRA_REBASELINE: [(90, True), (250, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(90, 2.5), (250, 1.0)],
            IS_EVENT: [(250, True)],
            IS_IMPROVEMENT_EVENT: [(250, True)],
            EVENT_TYPE: [(250, LABEL_IMPROVEMENT)],
            EVENT_SCORE: [(250, 1.0)],
            EVENT_REFERENCE_SCORE: [(250, 2.0)],
            EVENT_ID: [(250, 1.0)],
            IMPROVEMENT_EVENT_ID: [(250, 1.0)],
            DAYS_TO_NEXT_RELAPSE: [(0, 40)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (60, 20),
                (90, 50),
                (150, 110),
                (220, 180),
                (250, 210),
                (310, 270),
            ],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 30,
        },
    ), "Test 1 symmetric mode failed!"


# Does the roving descend to general?
def test_roving_pira_descends_to_general():
    assert raw_pira_progression_result_is_equal_to_target(
        relapse_timestamps=[15],
        follow_up_dataframe=pd.DataFrame(
            {
                TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
                EDSS_SCORE: [2.0, 1.0, 3.5, 3.0, 1.0, 2.0, 3.5],
            }
        ),
        targets_dict={
            DAYS_TO_NEXT_RELAPSE: [(0, 15), (10, 5)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (20, 5),
                (30, 15),
                (40, 25),
                (50, 35),
                (60, 45),
            ],
            IS_POST_EVENT_REBASELINE: [],
            IS_GENERAL_REBASELINE: [],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [],
            IS_PIRA_REBASELINE: [(20, True), (30, True), (40, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (20, 3.5),
                (30, 3.0),
                (40, 2.0),
            ],
            IS_POST_RELAPSE_REBASELINE: [(20, True)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_raw_after_relapse_max_time": 3,
            "opt_raw_before_relapse_max_time": 3,
        },
    ), "Roving baseline for RAW/PIRA is bugged..."
    assert raw_pira_progression_result_is_equal_to_target(
        relapse_timestamps=[15],
        follow_up_dataframe=pd.DataFrame(
            {
                TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
                EDSS_SCORE: [2.0, 1.0, 3.5, 3.0, 1.0, 1.5, 3.5],
            }
        ),
        targets_dict={
            DAYS_TO_NEXT_RELAPSE: [(0, 15), (10, 5)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (20, 5),
                (30, 15),
                (40, 25),
                (50, 35),
                (60, 45),
            ],
            IS_POST_EVENT_REBASELINE: [],
            IS_GENERAL_REBASELINE: [(40, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(40, 1.5)],
            IS_PIRA_REBASELINE: [(20, True), (30, True), (40, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
                (20, 3.5),
                (30, 3.0),
                (40, 1.5),
            ],
            IS_POST_RELAPSE_REBASELINE: [(20, True)],
        },
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": True,
            "opt_roving_reference_confirmation_time": 0.5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_raw_after_relapse_max_time": 3,
            "opt_raw_before_relapse_max_time": 3,
        },
    ), "Roving baseline for RAW/PIRA is bugged..."


# Event type assignment unconfirmed
def test_raw_pira_unconfirmed():
    test_dataframe = pd.DataFrame(
        {"days_after_baseline": [0, 20, 40], "edss_score": [1, 2.0, 2.0]}
    )
    # Test case 1 - no relapses
    test_case_1_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [],
        DAYS_TO_NEXT_RELAPSE: [],
        IS_POST_EVENT_REBASELINE: [(20, True)],
        IS_GENERAL_REBASELINE: [(20, True)],
        IS_PIRA_REBASELINE: [(20, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
        IS_EVENT: [(20, True)],
        IS_ACCRUAL_EVENT: [(20, True)],
        EVENT_TYPE: [(20, LABEL_PIRA)],
        EVENT_SCORE: [(20, 2.0)],
        EVENT_REFERENCE_SCORE: [(20, 1.0)],
        EVENT_ID: [(20, 1)],
        ACCRUAL_EVENT_ID: [(20, 1)],
        IS_POST_RELAPSE_REBASELINE: [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[],
        targets_dict=test_case_1_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
        },
    ), "Test 1 failed!"

    # Test cases 2 and 3 - relapse shortly before progression
    # The goal of these cases is to test the correct comparator sign (> vs. >=)
    test_case_2_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(20, 5), (40, 25)],
        DAYS_TO_NEXT_RELAPSE: [(0, 15)],
        IS_POST_EVENT_REBASELINE: [(20, True)],
        IS_GENERAL_REBASELINE: [(20, True)],
        IS_PIRA_REBASELINE: [(20, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
        IS_EVENT: [(20, True)],
        IS_ACCRUAL_EVENT: [(20, True)],
        EVENT_TYPE: [(20, LABEL_UNDEFINED)],
        EVENT_SCORE: [(20, 2.0)],
        EVENT_REFERENCE_SCORE: [(20, 1.0)],
        EVENT_ID: [(20, 1)],
        ACCRUAL_EVENT_ID: [(20, 1)],
        IS_POST_RELAPSE_REBASELINE: [(20, True)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict=test_case_2_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 4,
        },
    ), "Test 2 failed!"
    test_case_3_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(20, 5), (40, 25)],
        DAYS_TO_NEXT_RELAPSE: [(0, 15)],
        IS_POST_EVENT_REBASELINE: [(20, True)],
        IS_GENERAL_REBASELINE: [(20, True)],
        IS_PIRA_REBASELINE: [(20, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
        IS_EVENT: [(20, True)],
        IS_ACCRUAL_EVENT: [(20, True)],
        EVENT_TYPE: [(20, LABEL_RAW)],
        EVENT_SCORE: [(20, 2.0)],
        EVENT_REFERENCE_SCORE: [(20, 1.0)],
        EVENT_ID: [(20, 1)],
        ACCRUAL_EVENT_ID: [(20, 1)],
        IS_POST_RELAPSE_REBASELINE: [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict=test_case_3_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 5,
        },
    ), "Test 3 failed!"

    # Test cases 4 and 5 - relapse shortly after progression
    # The goal of these cases is to test the correct comparator sign (< vs. <=)
    test_case_4_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(40, 15)],
        DAYS_TO_NEXT_RELAPSE: [(0, 25), (20, 5)],
        IS_POST_EVENT_REBASELINE: [(20, True)],
        IS_GENERAL_REBASELINE: [(20, True)],
        IS_PIRA_REBASELINE: [(20, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
        IS_EVENT: [(20, True)],
        IS_ACCRUAL_EVENT: [(20, True)],
        EVENT_TYPE: [(20, LABEL_PIRA)],
        EVENT_SCORE: [(20, 2.0)],
        EVENT_REFERENCE_SCORE: [(20, 1.0)],
        EVENT_ID: [(20, 1)],
        ACCRUAL_EVENT_ID: [(20, 1)],
        IS_POST_RELAPSE_REBASELINE: [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[25],
        targets_dict=test_case_4_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 4,
            "opt_raw_after_relapse_max_time": 5,
        },
    ), "Test 4 failed!"
    test_case_5_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(40, 15)],
        DAYS_TO_NEXT_RELAPSE: [(0, 25), (20, 5)],
        IS_POST_EVENT_REBASELINE: [(20, True)],
        IS_GENERAL_REBASELINE: [(20, True)],
        IS_PIRA_REBASELINE: [(20, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 2.0)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 2.0)],
        IS_EVENT: [(20, True)],
        IS_ACCRUAL_EVENT: [(20, True)],
        EVENT_TYPE: [(20, LABEL_RAW)],
        EVENT_SCORE: [(20, 2.0)],
        EVENT_REFERENCE_SCORE: [(20, 1.0)],
        EVENT_ID: [(20, 1)],
        ACCRUAL_EVENT_ID: [(20, 1)],
        IS_POST_RELAPSE_REBASELINE: [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[25],
        targets_dict=test_case_5_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 5,
        },
    ), "Test 5 failed!"


# Event type assignment confirmed
def test_raw_pira_confirmed():
    test_dataframe = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60, 70, 80],
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.0, 3.0, 2.5, 2.5, 3.0],
        }
    )
    test_relapse_timestamps = [45]

    # Test case 1 - unconfirmed
    test_case_1_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(50, 5), (60, 15), (70, 25), (80, 35)],
        DAYS_TO_NEXT_RELAPSE: [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        IS_POST_EVENT_REBASELINE: [(30, True), (50, True)],
        IS_GENERAL_REBASELINE: [(30, True), (50, True)],
        IS_PIRA_REBASELINE: [(30, True), (50, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0), (50, 3.0)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (50, 3.0)],
        IS_EVENT: [(30, True), (50, True)],
        IS_ACCRUAL_EVENT: [(30, True), (50, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (50, LABEL_RAW)],
        EVENT_SCORE: [(30, 2.0), (50, 3.0)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (50, 2.0)],
        EVENT_ID: [(30, 1), (50, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (50, 2)],
        IS_POST_RELAPSE_REBASELINE: [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_1_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0,
            "opt_confirmation_included_values": "all",
            "opt_pira_allow_relapses_between_event_and_confirmation": False,
        },
    ), "Test 1 failed!"

    # Test case 2 - confirmation score in RAW window
    test_case_2_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(50, 5), (60, 15), (70, 25), (80, 35)],
        DAYS_TO_NEXT_RELAPSE: [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        IS_POST_EVENT_REBASELINE: [(30, True)],
        IS_GENERAL_REBASELINE: [(30, True)],
        IS_PIRA_REBASELINE: [(30, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (60, 2.5)],
        IS_EVENT: [(30, True)],
        IS_ACCRUAL_EVENT: [(30, True)],
        EVENT_TYPE: [(30, LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW)],
        EVENT_SCORE: [(30, 2.0)],
        EVENT_REFERENCE_SCORE: [(30, 1.0)],
        EVENT_ID: [(30, 1)],
        ACCRUAL_EVENT_ID: [(30, 1)],
        IS_POST_RELAPSE_REBASELINE: [(60, True)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_2_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_included_values": "all",
            "opt_pira_allow_relapses_between_event_and_confirmation": False,
        },
    ), "Test 2 failed!"

    # Test case 3 - confirmation score before RAW window
    test_case_3_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(50, 5), (60, 15), (70, 25), (80, 35)],
        DAYS_TO_NEXT_RELAPSE: [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        IS_POST_EVENT_REBASELINE: [(30, True)],
        IS_GENERAL_REBASELINE: [(30, True)],
        IS_PIRA_REBASELINE: [(30, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (60, 2.5)],
        IS_EVENT: [(30, True)],
        IS_ACCRUAL_EVENT: [(30, True)],
        EVENT_TYPE: [(30, LABEL_PIRA)],
        EVENT_SCORE: [(30, 2.0)],
        EVENT_REFERENCE_SCORE: [(30, 1.0)],
        EVENT_ID: [(30, 1)],
        ACCRUAL_EVENT_ID: [(30, 1)],
        IS_POST_RELAPSE_REBASELINE: [(60, True)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_3_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_time": 4,
            "opt_raw_after_relapse_max_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 10,
            "opt_confirmation_included_values": "all",
            "opt_pira_allow_relapses_between_event_and_confirmation": False,
        },
    ), "Test 3 failed!"
    # Test case 4 - (2, 10, True, 30, "all", True, "Relapse within confirmation interval")
    test_case_4_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(50, 5), (60, 15), (70, 25), (80, 35)],
        DAYS_TO_NEXT_RELAPSE: [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        IS_POST_EVENT_REBASELINE: [(30, True)],
        IS_GENERAL_REBASELINE: [(30, True)],
        IS_PIRA_REBASELINE: [(30, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (60, 2.5)],
        IS_EVENT: [(30, True)],
        IS_ACCRUAL_EVENT: [(30, True)],
        EVENT_TYPE: [(30, LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW)],
        EVENT_SCORE: [(30, 2.0)],
        EVENT_REFERENCE_SCORE: [(30, 1.0)],
        EVENT_ID: [(30, 1)],
        ACCRUAL_EVENT_ID: [(30, 1)],
        IS_POST_RELAPSE_REBASELINE: [(60, True)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_4_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_time": 2,
            "opt_raw_after_relapse_max_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "all",
            "opt_pira_allow_relapses_between_event_and_confirmation": False,
        },
    ), "Test 4 failed!"
    # Test case 5 - (2, 10, True, 30, "last", True, "Last only, relapses in confirmation allowed")
    test_case_5_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(50, 5), (60, 15), (70, 25), (80, 35)],
        DAYS_TO_NEXT_RELAPSE: [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        IS_POST_EVENT_REBASELINE: [(30, True), (50, True)],
        IS_GENERAL_REBASELINE: [(30, True), (50, True)],
        IS_PIRA_REBASELINE: [(30, True), (50, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0), (50, 3.0)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (50, 3.0)],
        IS_EVENT: [(30, True), (50, True)],
        IS_ACCRUAL_EVENT: [(30, True), (50, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (50, LABEL_RAW)],
        EVENT_SCORE: [(30, 2.0), (50, 3.0)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (50, 2.0)],
        EVENT_ID: [(30, 1), (50, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (50, 2)],
        IS_POST_RELAPSE_REBASELINE: [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_5_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_time": 2,
            "opt_raw_after_relapse_max_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "last",
            "opt_pira_allow_relapses_between_event_and_confirmation": True,
        },
    ), "Test 5 failed!"
    # Test case 6 - (2, 10, True, 30, "last", False, "Last only, relapses in confirmation interval not allowed")
    test_case_6_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(50, 5), (60, 15), (70, 25), (80, 35)],
        DAYS_TO_NEXT_RELAPSE: [(0, 45), (10, 35), (20, 25), (30, 15), (40, 5)],
        IS_POST_EVENT_REBASELINE: [(30, True), (50, True)],
        IS_GENERAL_REBASELINE: [(30, True), (50, True)],
        IS_PIRA_REBASELINE: [(30, True), (50, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0), (50, 3.0)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 2.0), (50, 3.0)],
        IS_EVENT: [(30, True), (50, True)],
        IS_ACCRUAL_EVENT: [(30, True), (50, True)],
        EVENT_TYPE: [(30, LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW), (50, LABEL_RAW)],
        EVENT_SCORE: [(30, 2.0), (50, 3.0)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (50, 2.0)],
        EVENT_ID: [(30, 1), (50, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (50, 2)],
        IS_POST_RELAPSE_REBASELINE: [],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_6_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_time": 2,
            "opt_raw_after_relapse_max_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 30,
            "opt_confirmation_included_values": "last",
            "opt_pira_allow_relapses_between_event_and_confirmation": False,
        },
    ), "Test 6 failed!"
    # Test case 7 - undefined candidate lower than PIRA reference is event
    assert raw_pira_progression_result_is_equal_to_target(
        relapse_timestamps=[25, 39],
        follow_up_dataframe=pd.DataFrame(
            {
                TIMESTAMP: [0, 10, 20, 30, 40, 50, 60, 70],
                EDSS_SCORE: [6.0, 6.5, 7.0, 7.0, 6.0, 6.5, 7.0, 7.0],
            }
        ),
        targets_dict={
            DAYS_TO_NEXT_RELAPSE: [(0, 25), (10, 15), (20, 5), (30, 9)],
            DAYS_SINCE_PREVIOUS_RELAPSE: [
                (30, 5),
                (40, 1),
                (50, 11),
                (60, 21),
                (70, 31),
            ],
            IS_POST_EVENT_REBASELINE: [(50, True), (60, True)],
            IS_GENERAL_REBASELINE: [(50, True), (60, True)],
            EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(50, 6.5), (60, 7)],
            IS_PIRA_REBASELINE: [(30, True), (50, True), (60, True)],
            EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 7), (50, 6.5), (60, 7)],
            IS_POST_RELAPSE_REBASELINE: [(30, True), (50, True)],
            IS_EVENT: [(50, True), (60, True)],
            IS_ACCRUAL_EVENT: [(50, True), (60, True)],
            EVENT_TYPE: [(50, LABEL_UNDEFINED), (60, LABEL_PIRA)],
            EVENT_SCORE: [(50, 6.5), (60, 7)],
            EVENT_REFERENCE_SCORE: [(50, 6), (60, 6.5)],
            EVENT_ID: [(50, 1), (60, 2)],
            ACCRUAL_EVENT_ID: [(50, 1), (60, 2)],
        },
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": -1,
            "opt_raw_after_relapse_max_time": 3,
            "opt_raw_before_relapse_max_time": 3,
        },
    ), "Test 7 failed!"


def test_post_relapse_rebaselining_higher_equal_lower():
    test_relapse_timestamps = [15]
    common_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(20, 5), (30, 15), (40, 25), (50, 35)],
        DAYS_TO_NEXT_RELAPSE: [(0, 15), (10, 5)],
    }

    # Test case 1 - stable baseline
    test_case_1_dataframe = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50],
            EDSS_SCORE: [2.5, 3, 3, 2.5, 2.5, 2.5],
        }
    )
    test_case_1_targets = {}
    test_case_1_targets = {**test_case_1_targets, **common_targets}
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_case_1_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_1_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 1 failed!"

    # Test case 2 - residual disability
    test_case_2_dataframe = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50],
            EDSS_SCORE: [2.5, 3, 3, 3, 3, 3],
        }
    )
    test_case_2_targets = {
        IS_POST_EVENT_REBASELINE: [],
        IS_PIRA_REBASELINE: [(30, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 3)],
        IS_POST_RELAPSE_REBASELINE: [(30, True)],
    }
    test_case_2_targets = {**test_case_2_targets, **common_targets}
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_case_2_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_2_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 2 failed!"

    # Test case 3 - lower score after relapse
    test_case_3_dataframe = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50],
            EDSS_SCORE: [2.5, 3, 3, 2, 2, 2],
        }
    )
    test_case_3_targets = {}
    test_case_3_targets = {**test_case_3_targets, **common_targets}
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_case_3_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_3_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 3 failed!"

    # Test case 4 - event
    test_case_4_dataframe = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50],
            EDSS_SCORE: [2.5, 3.5, 3.5, 3, 3, 3],
        }
    )
    test_case_4_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(20, 5), (30, 15), (40, 25), (50, 35)],
        DAYS_TO_NEXT_RELAPSE: [(0, 15), (10, 5)],
        IS_POST_EVENT_REBASELINE: [(10, True)],
        IS_GENERAL_REBASELINE: [(10, True)],
        IS_PIRA_REBASELINE: [(10, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 3.5)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 3.5)],
        IS_EVENT: [(10, True)],
        IS_ACCRUAL_EVENT: [(10, True)],
        EVENT_TYPE: [(10, LABEL_RAW)],
        EVENT_SCORE: [(10, 3.5)],
        EVENT_REFERENCE_SCORE: [(10, 2.5)],
        EVENT_ID: [(10, 1)],
        ACCRUAL_EVENT_ID: [(10, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_case_4_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_4_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 4 failed!"


# With backtracking
def test_post_relapse_rebaselining_backtracking():
    test_relapse_timestamps = [25]
    test_dataframe = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60],
            EDSS_SCORE: [3.5, 3, 2.5, 3, 4, 4, 4],
        }
    )

    # Test case 1 - with backtracking, stopped by post-relapse re-baselining
    test_case_1_targets = {
        IS_POST_EVENT_REBASELINE: [(40, True)],
        IS_PIRA_REBASELINE: [(10, True), (20, True), (30, True), (40, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (10, 3),
            (20, 2.5),
            (30, 3),
            (40, 4),
        ],
        IS_POST_RELAPSE_REBASELINE: [(30, True)],
        DAYS_SINCE_PREVIOUS_RELAPSE: [(30, 5), (40, 15), (50, 25), (60, 35)],
        DAYS_TO_NEXT_RELAPSE: [(0, 25), (10, 15), (20, 5)],
        IS_GENERAL_REBASELINE: [(10, True), (20, True), (40, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 3), (20, 2.5), (40, 4)],
        IS_EVENT: [(40, True)],
        IS_ACCRUAL_EVENT: [(40, True)],
        EVENT_TYPE: [(40, LABEL_UNDEFINED)],
        EVENT_SCORE: [(40, 4)],
        EVENT_REFERENCE_SCORE: [(40, 3.0)],
        EVENT_ID: [(40, 1)],
        ACCRUAL_EVENT_ID: [(40, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_1_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 3,
            "opt_raw_after_relapse_max_time": 3,
            "opt_require_confirmation": False,
            "opt_minimal_distance_time": 25,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 1 failed!"

    # Test case 2 - no residual disability after relapse
    test_case_2_targets = {
        IS_POST_EVENT_REBASELINE: [(40, True)],
        IS_PIRA_REBASELINE: [(10, True), (20, True), (40, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(10, 3), (20, 2.5), (40, 4)],
        IS_POST_RELAPSE_REBASELINE: [],
        DAYS_SINCE_PREVIOUS_RELAPSE: [(30, 5), (40, 15), (50, 25), (60, 35)],
        DAYS_TO_NEXT_RELAPSE: [(0, 25), (10, 15), (20, 5)],
        IS_GENERAL_REBASELINE: [(10, True), (20, True), (40, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(10, 3), (20, 2.5), (40, 4)],
        IS_EVENT: [(40, True)],
        IS_ACCRUAL_EVENT: [(40, True)],
        EVENT_TYPE: [(40, LABEL_PIRA)],
        EVENT_SCORE: [(40, 4)],
        EVENT_REFERENCE_SCORE: [(40, 3)],
        EVENT_ID: [(40, 1)],
        ACCRUAL_EVENT_ID: [(40, 1)],
    }
    test_dataframe.at[3, "edss_score"] = 2.5
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict=test_case_2_targets,
        args_dict={
            "annotation_mode": ACCRUAL_MODE_NAME,
            "opt_baseline_type": "roving",
            "opt_roving_reference_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 3,
            "opt_raw_after_relapse_max_time": 3,
            "opt_require_confirmation": False,
            "opt_minimal_distance_time": 25,
            "opt_minimal_distance_type": "reference",
            "opt_minimal_distance_backtrack_decrease": True,
        },
    ), "Test 2 failed!"


def test_post_relapse_rebaselining_with_missing_and_overlapping():
    test_relapses_cases_1_2_3_4 = [6, 30]
    test_dataframe_cases_1_2_3_4 = pd.DataFrame(
        {TIMESTAMP: [0, 8, 36, 48, 60], EDSS_SCORE: [1, 1.5, 2.5, 3, 3.5]}
    )
    test_common_targets_cases_1_2_3_4 = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(8, 2), (36, 6), (48, 18), (60, 30)],
        DAYS_TO_NEXT_RELAPSE: [(0, 6), (8, 22)],
    }

    # Test case 1 - assessments well-separated
    test_case_1_targets = {
        IS_POST_EVENT_REBASELINE: [(36, True), (60, True)],
        IS_PIRA_REBASELINE: [(8, True), (36, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(8, 1.5), (36, 2.5), (60, 3.5)],
        IS_POST_RELAPSE_REBASELINE: [(8, True), (36, True)],
        IS_GENERAL_REBASELINE: [(36, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(36, 2.5), (60, 3.5)],
        IS_EVENT: [(36, True), (60, True)],
        IS_ACCRUAL_EVENT: [(36, True), (60, True)],
        EVENT_TYPE: [(36, LABEL_UNDEFINED), (60, LABEL_PIRA)],
        EVENT_SCORE: [(36, 2.5), (60, 3.5)],
        EVENT_REFERENCE_SCORE: [(36, 1.0), (60, 2.5)],
        EVENT_ID: [(36, 1), (60, 2)],
        ACCRUAL_EVENT_ID: [(36, 1), (60, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_1_2_3_4,
        relapse_timestamps=test_relapses_cases_1_2_3_4,
        targets_dict={**test_case_1_targets, **test_common_targets_cases_1_2_3_4},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 2,
            "opt_raw_after_relapse_max_time": 1,
        },
    ), "Test 1 failed!"

    # Test case 2 - rebaseline of first relapse after second
    # relapse, within buffer
    test_case_2_targets = {
        IS_POST_EVENT_REBASELINE: [(36, True), (60, True)],
        IS_PIRA_REBASELINE: [(36, True), (48, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(36, 2.5), (48, 3.0), (60, 3.5)],
        IS_POST_RELAPSE_REBASELINE: [(36, True), (48, True)],
        IS_GENERAL_REBASELINE: [(36, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(36, 2.5), (60, 3.5)],
        IS_EVENT: [(36, True), (60, True)],
        IS_ACCRUAL_EVENT: [(36, True), (60, True)],
        EVENT_TYPE: [(36, LABEL_UNDEFINED), (60, LABEL_UNDEFINED)],
        EVENT_SCORE: [(36, 2.5), (60, 3.5)],
        EVENT_REFERENCE_SCORE: [(36, 1.0), (60, 2.5)],
        EVENT_ID: [(36, 1), (60, 2)],
        ACCRUAL_EVENT_ID: [(36, 1), (60, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_1_2_3_4,
        relapse_timestamps=test_relapses_cases_1_2_3_4,
        targets_dict={**test_case_2_targets, **test_common_targets_cases_1_2_3_4},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 4,
            "opt_raw_after_relapse_max_time": 12,
        },
    ), "Test 2 failed!"

    # Test case 3 - rebaseline of first relapse after second relapse,
    # after second buffer (same for both)
    test_case_3_targets = {
        IS_POST_EVENT_REBASELINE: [(36, True), (60, True)],
        IS_PIRA_REBASELINE: [(36, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(36, 2.5), (60, 3.5)],
        IS_POST_RELAPSE_REBASELINE: [(36, True)],
        IS_GENERAL_REBASELINE: [(36, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(36, 2.5), (60, 3.5)],
        IS_EVENT: [(36, True), (60, True)],
        IS_ACCRUAL_EVENT: [(36, True), (60, True)],
        EVENT_TYPE: [(36, LABEL_UNDEFINED), (60, LABEL_PIRA)],
        EVENT_SCORE: [(36, 2.5), (60, 3.5)],
        EVENT_REFERENCE_SCORE: [(36, 1.0), (60, 2.5)],
        EVENT_ID: [(36, 1), (60, 2)],
        ACCRUAL_EVENT_ID: [(36, 1), (60, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_1_2_3_4,
        relapse_timestamps=test_relapses_cases_1_2_3_4,
        targets_dict={**test_case_3_targets, **test_common_targets_cases_1_2_3_4},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 4,
            "opt_raw_after_relapse_max_time": 4,
        },
    ), "Test 3 failed!"

    # Test case 4 - overlapping RAW windows
    test_case_4_targets = {
        IS_POST_EVENT_REBASELINE: [(36, True), (60, True)],
        IS_PIRA_REBASELINE: [(36, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(36, 2.5), (60, 3.5)],
        IS_POST_RELAPSE_REBASELINE: [(60, True)],
        IS_GENERAL_REBASELINE: [(36, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(36, 2.5), (60, 3.5)],
        IS_EVENT: [(36, True), (60, True)],
        IS_ACCRUAL_EVENT: [(36, True), (60, True)],
        EVENT_TYPE: [(36, LABEL_RAW), (60, LABEL_UNDEFINED)],
        EVENT_SCORE: [(36, 2.5), (60, 3.5)],
        EVENT_REFERENCE_SCORE: [(36, 1.0), (60, 2.5)],
        EVENT_ID: [(36, 1), (60, 2)],
        ACCRUAL_EVENT_ID: [(36, 1), (60, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_1_2_3_4,
        relapse_timestamps=test_relapses_cases_1_2_3_4,
        targets_dict={**test_case_4_targets, **test_common_targets_cases_1_2_3_4},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 4,
            "opt_raw_after_relapse_max_time": 20,
        },
    ), "Test 4 failed!"

    # Test case 5 - rebaseline of first relapse before second, but within RAW window
    # TODO --- HERE events at? Print.
    test_relapses_cases_5 = [6, 30]
    test_dataframe_cases_5 = pd.DataFrame(
        {TIMESTAMP: [0, 8, 28, 48, 60], EDSS_SCORE: [1, 1.5, 2.5, 3, 3.5]}
    )
    test_case_5_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(8, 2), (28, 22), (48, 18), (60, 30)],
        DAYS_TO_NEXT_RELAPSE: [(0, 6), (8, 22), (28, 2)],
        IS_POST_EVENT_REBASELINE: [(28, True)],
        IS_PIRA_REBASELINE: [(28, True), (48, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(28, 2.5), (48, 3)],
        IS_POST_RELAPSE_REBASELINE: [(28, True), (48, True)],
        IS_GENERAL_REBASELINE: [(28, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(28, 2.5)],
        IS_EVENT: [(28, True)],
        IS_ACCRUAL_EVENT: [(28, True)],
        EVENT_TYPE: [(28, LABEL_UNDEFINED)],
        EVENT_SCORE: [(28, 2.5)],
        EVENT_REFERENCE_SCORE: [(28, 1.0)],
        EVENT_ID: [(28, 1)],
        ACCRUAL_EVENT_ID: [(28, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_5,
        relapse_timestamps=test_relapses_cases_5,
        targets_dict=test_case_5_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "undefined_events_annotation_mode": "re-baselining only",
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 4,
            "opt_raw_after_relapse_max_time": 4,
        },
    ), "Test 5 failed!"
    test_case_5b_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(8, 2), (28, 22), (48, 18), (60, 30)],
        DAYS_TO_NEXT_RELAPSE: [(0, 6), (8, 22), (28, 2)],
        IS_POST_EVENT_REBASELINE: [(28, True), (60, True)],
        IS_PIRA_REBASELINE: [(28, True), (48, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(28, 2.5), (48, 3), (60, 3.5)],
        IS_POST_RELAPSE_REBASELINE: [(28, True), (48, True)],
        IS_GENERAL_REBASELINE: [(28, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(28, 2.5), (60, 3.5)],
        IS_EVENT: [(28, True), (60, True)],
        IS_ACCRUAL_EVENT: [(28, True), (60, True)],
        EVENT_TYPE: [(28, LABEL_UNDEFINED), (60, LABEL_UNDEFINED)],
        EVENT_SCORE: [(28, 2.5), (60, 3.5)],
        EVENT_REFERENCE_SCORE: [(28, 1.0), (60, 2.5)],
        EVENT_ID: [(28, 1), (60, 2)],
        ACCRUAL_EVENT_ID: [(28, 1), (60, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_5,
        relapse_timestamps=test_relapses_cases_5,
        targets_dict=test_case_5b_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 4,
            "opt_raw_after_relapse_max_time": 4,
        },
    ), "Test 5b failed!"

    # Test case 6 - multiple non-overlapping and overlapping relapses
    test_relapses_cases_6 = [15, 25, 40, 48]
    test_dataframe_cases_6 = pd.DataFrame(
        {TIMESTAMP: [0, 10, 50, 60, 70], EDSS_SCORE: [1, 1, 2, 2, 2.5]}
    )
    test_case_6_targets = {
        DAYS_SINCE_PREVIOUS_RELAPSE: [(50, 2), (60, 12), (70, 22)],
        DAYS_TO_NEXT_RELAPSE: [(0, 15), (10, 5)],
        IS_POST_EVENT_REBASELINE: [(50, True)],
        IS_PIRA_REBASELINE: [(50, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(50, 2)],
        IS_POST_RELAPSE_REBASELINE: [(50, True)],
        IS_GENERAL_REBASELINE: [(50, True)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(50, 2)],
        IS_EVENT: [(50, True)],
        IS_ACCRUAL_EVENT: [(50, True)],
        EVENT_TYPE: [(50, LABEL_UNDEFINED)],
        EVENT_SCORE: [(50, 2)],
        EVENT_REFERENCE_SCORE: [(50, 1.0)],
        EVENT_ID: [(50, 1)],
        ACCRUAL_EVENT_ID: [(50, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_cases_6,
        relapse_timestamps=test_relapses_cases_6,
        targets_dict=test_case_6_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 2,
            "opt_raw_after_relapse_max_time": 7,
        },
    ), "Test 6 failed!"


def test_post_relapse_rebaselining_with_event():
    test_dataframe = pd.DataFrame(
        {TIMESTAMP: [0, 20, 40, 60], EDSS_SCORE: [1, 1.5, 2.5, 2]}
    )
    test_relapses = [15, 25]

    test_common_targets = {
        DAYS_TO_NEXT_RELAPSE: [(0, 15), (20, 5)],
        DAYS_SINCE_PREVIOUS_RELAPSE: [(20, 5), (40, 15), (60, 35)],
        IS_POST_EVENT_REBASELINE: [(40, True)],
        IS_PIRA_REBASELINE: [(20, True), (40, True)],
        IS_GENERAL_REBASELINE: [(40, True)],
        IS_POST_RELAPSE_REBASELINE: [(20, True), (40, True)],
        IS_EVENT: [(40, True)],
        IS_ACCRUAL_EVENT: [(40, True)],
        EVENT_TYPE: [(40, LABEL_UNDEFINED)],
        EVENT_ID: [(40, 1)],
        ACCRUAL_EVENT_ID: [(40, 1)],
    }
    # Test case 1 - event unconfirmed
    test_case_1_targets = {
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 1.5), (40, 2.5)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(40, 2.5)],
        EVENT_SCORE: [(40, 2.5)],
        EVENT_REFERENCE_SCORE: [(40, 1.0)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapses,
        targets_dict={**test_case_1_targets, **test_common_targets},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 4,
        },
    ), "Test 1 failed!"
    # Test case 2 - event confirmed
    test_case_2_targets = {
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 1.5), (40, 2)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(40, 2)],
        EVENT_SCORE: [(40, 2)],
        EVENT_REFERENCE_SCORE: [(40, 1.0)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapses,
        targets_dict={**test_case_2_targets, **test_common_targets},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 4,
        },
    ), "Test 2 failed!"


# No undefined events option
def test_undefined_progression_never_option():
    test_dataframe = pd.DataFrame(
        {
            TIMESTAMP: [0, 20, 40, 60, 80],
            EDSS_SCORE: [1, 1.5, 2.5, 3.5, 4.5],
        }
    )
    test_common_targets = {
        DAYS_TO_NEXT_RELAPSE: [(0, 15)],
        DAYS_SINCE_PREVIOUS_RELAPSE: [(20, 5), (40, 25), (60, 45), (80, 65)],
    }

    # Test case 1 - allow undefined at re-baselining, all events
    test_case_1_targets = {
        IS_POST_EVENT_REBASELINE: [(40, True), (60, True), (80, True)],
        IS_PIRA_REBASELINE: [(40, True), (60, True), (80, True)],
        IS_GENERAL_REBASELINE: [(40, True), (60, True), (80, True)],
        IS_POST_RELAPSE_REBASELINE: [(40, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(40, 2.5), (60, 3.5), (80, 4.5)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(40, 2.5), (60, 3.5), (80, 4.5)],
        IS_EVENT: [(40, True), (60, True), (80, True)],
        IS_ACCRUAL_EVENT: [(40, True), (60, True), (80, True)],
        EVENT_TYPE: [
            (40, LABEL_UNDEFINED),
            (60, LABEL_PIRA),
            (80, LABEL_PIRA),
        ],
        EVENT_SCORE: [(40, 2.5), (60, 3.5), (80, 4.5)],
        EVENT_REFERENCE_SCORE: [(40, 1.0), (60, 2.5), (80, 3.5)],
        EVENT_ID: [(40, 1), (60, 2), (80, 3)],
        ACCRUAL_EVENT_ID: [(40, 1), (60, 2), (80, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict={**test_common_targets, **test_case_1_targets},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "undefined_events_annotation_mode": "re-baselining only",
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
        },
    ), "Test 1 failed!"

    # Test case 2 - allow undefined at re-baselining, first event only
    test_case_2_targets = {
        IS_POST_RELAPSE_REBASELINE: [(40, True)],
        IS_EVENT: [(40, True)],
        IS_ACCRUAL_EVENT: [(40, True)],
        EVENT_TYPE: [(40, LABEL_UNDEFINED)],
        EVENT_SCORE: [(40, 2.5)],
        EVENT_REFERENCE_SCORE: [(40, 1.0)],
        EVENT_ID: [(40, 1)],
        ACCRUAL_EVENT_ID: [(40, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict={**test_common_targets, **test_case_2_targets},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "undefined_events_annotation_mode": "re-baselining only",
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "return_first_event_only": True,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
        },
    ), "Test 2 failed!"

    # Test case 3 - don't allow undefined at re-baselining, all events
    test_case_3_targets = {
        IS_POST_EVENT_REBASELINE: [(60, True), (80, True)],
        IS_PIRA_REBASELINE: [(40, True), (60, True), (80, True)],
        IS_GENERAL_REBASELINE: [(60, True), (80, True)],
        IS_POST_RELAPSE_REBASELINE: [(40, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(40, 2.5), (60, 3.5), (80, 4.5)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(60, 3.5), (80, 4.5)],
        IS_EVENT: [(60, True), (80, True)],
        IS_ACCRUAL_EVENT: [(60, True), (80, True)],
        EVENT_TYPE: [(60, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(60, 3.5), (80, 4.5)],
        EVENT_REFERENCE_SCORE: [(60, 2.5), (80, 3.5)],
        EVENT_ID: [(60, 1), (80, 2)],
        ACCRUAL_EVENT_ID: [(60, 1), (80, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict={**test_common_targets, **test_case_3_targets},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "undefined_events_annotation_mode": "never",
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "return_first_event_only": False,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
        },
    ), "Test 3 failed!"

    # Test case 4 - don't allow undefined at re-baselining, last event only
    test_case_4_targets = {
        IS_PIRA_REBASELINE: [(40, True)],
        IS_POST_RELAPSE_REBASELINE: [(40, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(40, 2.5)],
        IS_EVENT: [(60, True)],
        IS_ACCRUAL_EVENT: [(60, True)],
        EVENT_TYPE: [(60, LABEL_PIRA)],
        EVENT_SCORE: [(60, 3.5)],
        EVENT_REFERENCE_SCORE: [(60, 2.5)],
        EVENT_ID: [(60, 1)],
        ACCRUAL_EVENT_ID: [(60, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[15],
        targets_dict={**test_common_targets, **test_case_4_targets},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "undefined_events_annotation_mode": "never",
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "return_first_event_only": True,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
        },
    ), "Test 4 failed!"

    with_confirmation_test_dataframe = pd.DataFrame(
        {TIMESTAMP: [0, 20, 40, 60], EDSS_SCORE: [1, 1.5, 2.5, 2]}
    )
    with_confirmation_test_common_targets = {
        DAYS_TO_NEXT_RELAPSE: [(0, 15), (20, 5)],
        DAYS_SINCE_PREVIOUS_RELAPSE: [(20, 5), (40, 15), (60, 35)],
    }
    # Test case 5 - allow undefined at re-baselining, all events
    test_case_5_targets = {
        IS_POST_EVENT_REBASELINE: [(40, True)],
        IS_PIRA_REBASELINE: [(20, True), (40, True)],
        IS_GENERAL_REBASELINE: [(40, True)],
        IS_POST_RELAPSE_REBASELINE: [(20, True), (40, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 1.5), (40, 2.0)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(40, 2.0)],
        IS_EVENT: [(40, True)],
        IS_ACCRUAL_EVENT: [(40, True)],
        EVENT_TYPE: [(40, LABEL_UNDEFINED)],
        EVENT_SCORE: [(40, 2.0)],
        EVENT_REFERENCE_SCORE: [(40, 1.0)],
        EVENT_ID: [(40, 1)],
        ACCRUAL_EVENT_ID: [(40, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=with_confirmation_test_dataframe,
        relapse_timestamps=[15, 25],
        targets_dict={**with_confirmation_test_common_targets, **test_case_5_targets},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "undefined_events_annotation_mode": "re-baselining only",
            "opt_baseline_type": "fixed",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 5 failed!"
    # Test case 6 - do not allow undefined at re-baselining, all events
    test_case_6_targets = {
        IS_PIRA_REBASELINE: [(20, True), (40, True)],
        IS_POST_RELAPSE_REBASELINE: [(20, True), (40, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 1.5), (40, 2.5)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=with_confirmation_test_dataframe,
        relapse_timestamps=[15, 25],
        targets_dict={**with_confirmation_test_common_targets, **test_case_6_targets},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "undefined_events_annotation_mode": "never",
            "opt_baseline_type": "fixed",
            "return_first_event_only": False,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 4,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 6 failed!"
    # Test case 7 - 'never' option without relapses
    test_case_7_targets = {
        IS_POST_EVENT_REBASELINE: [(40, True), (60, True), (80, True)],
        IS_PIRA_REBASELINE: [(40, True), (60, True), (80, True)],
        IS_GENERAL_REBASELINE: [(40, True), (60, True), (80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(40, 2.5), (60, 3.5), (80, 4.5)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(40, 2.5), (60, 3.5), (80, 4.5)],
        IS_EVENT: [(40, True), (60, True), (80, True)],
        IS_ACCRUAL_EVENT: [(40, True), (60, True), (80, True)],
        EVENT_TYPE: [(40, LABEL_PIRA), (60, LABEL_PIRA), (80, LABEL_PIRA)],
        EVENT_SCORE: [(40, 2.5), (60, 3.5), (80, 4.5)],
        EVENT_REFERENCE_SCORE: [(40, 1.0), (60, 2.5), (80, 3.5)],
        EVENT_ID: [(40, 1), (60, 2), (80, 3)],
        ACCRUAL_EVENT_ID: [(40, 1), (60, 2), (80, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=[],
        targets_dict=test_case_7_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "undefined_events_annotation_mode": "never",
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": False,
            "return_first_event_only": False,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
        },
    ), "Test 7 failed!"


def test_undefined_event_annotation_options():
    # TODO: more cases...
    test_dataframe = pd.DataFrame(
        {
            TIMESTAMP: [0, 60, 90, 150, 220, 250, 310, 400, 460],
            EDSS_SCORE: [1.0, 1.5, 2.0, 2.0, 3.5, 2.5, 3.0, 3.5, 4.0],
        }
    )
    test_relapse_timestamps = [40, 200]

    test_common_targets = {
        DAYS_TO_NEXT_RELAPSE: [(0, 40), (60, 140), (90, 110), (150, 50)],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (60, 20),
            (90, 50),
            (150, 110),
            (220, 20),
            (250, 50),
            (310, 110),
            (400, 200),
            (460, 260),
        ],
    }

    # Test case 1 - allow undefined at all positions
    test_case_1_targets = {
        IS_POST_EVENT_REBASELINE: [(90, True), (310, True)],
        IS_PIRA_REBASELINE: [(90, True), (250, True), (310, True)],
        IS_GENERAL_REBASELINE: [(90, True), (310, True)],
        IS_POST_RELAPSE_REBASELINE: [(90, True), (250, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(90, 2.0), (250, 2.5), (310, 3.0)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(90, 2.0), (310, 3.0)],
        IS_EVENT: [(90, True), (310, True)],
        IS_ACCRUAL_EVENT: [(90, True), (310, True)],
        EVENT_TYPE: [
            (90, LABEL_UNDEFINED),
            (310, LABEL_UNDEFINED),
        ],
        EVENT_SCORE: [(90, 2.0), (310, 3.0)],
        EVENT_REFERENCE_SCORE: [(90, 1.0), (310, 2.0)],
        EVENT_ID: [(90, 1), (310, 2)],
        ACCRUAL_EVENT_ID: [(90, 1), (310, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict={**test_common_targets, **test_case_1_targets},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "undefined_events_annotation_mode": "all",
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 30,
        },
    ), "Test 1 failed!"
    # Test case 2 - allow undefined at all positions
    test_case_2_targets = {
        IS_POST_EVENT_REBASELINE: [(90, True), (400, True)],
        IS_PIRA_REBASELINE: [(90, True), (250, True), (400, True)],
        IS_GENERAL_REBASELINE: [(90, True), (400, True)],
        IS_POST_RELAPSE_REBASELINE: [(90, True), (250, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(90, 2.0), (250, 2.5), (400, 3.5)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(90, 2.0), (400, 3.5)],
        IS_EVENT: [(90, True), (400, True)],
        IS_ACCRUAL_EVENT: [(90, True), (400, True)],
        EVENT_TYPE: [
            (90, LABEL_UNDEFINED),
            (400, LABEL_PIRA),
        ],
        EVENT_SCORE: [(90, 2.0), (400, 3.5)],
        EVENT_REFERENCE_SCORE: [(90, 1.0), (400, 2.5)],
        EVENT_ID: [(90, 1), (400, 2)],
        ACCRUAL_EVENT_ID: [(90, 1), (400, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict={**test_common_targets, **test_case_2_targets},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "undefined_events_annotation_mode": "re-baselining only",
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 30,
        },
    ), "Test 2 failed!"
    # Test case 3 - don't allow undefined
    test_case_3_targets = {
        IS_POST_EVENT_REBASELINE: [(220, True), (400, True)],
        IS_PIRA_REBASELINE: [(90, True), (220, True), (400, True)],
        IS_GENERAL_REBASELINE: [(220, True), (400, True)],
        IS_POST_RELAPSE_REBASELINE: [(90, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(90, 2.0), (220, 2.5), (400, 3.5)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(220, 2.5), (400, 3.5)],
        IS_EVENT: [(220, True), (400, True)],
        IS_ACCRUAL_EVENT: [(220, True), (400, True)],
        EVENT_TYPE: [
            (220, LABEL_RAW),
            (400, LABEL_PIRA),
        ],
        EVENT_SCORE: [(220, 2.5), (400, 3.5)],
        EVENT_REFERENCE_SCORE: [(220, 1.0), (400, 2.5)],
        EVENT_ID: [(220, 1), (400, 2)],
        ACCRUAL_EVENT_ID: [(220, 1), (400, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe,
        relapse_timestamps=test_relapse_timestamps,
        targets_dict={**test_common_targets, **test_case_3_targets},
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "undefined_events_annotation_mode": "never",
            "opt_baseline_type": "fixed",
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
            "opt_raw_before_relapse_max_time": 30,
            "opt_raw_after_relapse_max_time": 30,
        },
    ), "Test 3 failed!"


def test_event_merging_with_relapses():
    ### Part 1 - from old version ###
    # Test case 1 - undefined must remain singular
    test_dataframe_case_1 = pd.DataFrame(
        {TIMESTAMP: [0, 8, 36, 48, 60], EDSS_SCORE: [1, 1.5, 2.5, 3, 3.5]}
    )

    test_case_1_targets = {
        DAYS_TO_NEXT_RELAPSE: [(0, 6), (8, 22)],
        DAYS_SINCE_PREVIOUS_RELAPSE: [(8, 2), (36, 6), (48, 18), (60, 30)],
        IS_POST_EVENT_REBASELINE: [(36, True), (60, True)],
        IS_PIRA_REBASELINE: [(36, True), (60, True)],
        IS_GENERAL_REBASELINE: [(36, True), (60, True)],
        IS_POST_RELAPSE_REBASELINE: [(36, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(36, 2.5), (60, 3.5)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(36, 2.5), (60, 3.5)],
        IS_EVENT: [(36, True), (60, True)],
        IS_ACCRUAL_EVENT: [(36, True), (60, True)],
        EVENT_TYPE: [(36, LABEL_UNDEFINED), (60, LABEL_PIRA)],
        EVENT_SCORE: [(36, 2.5), (60, 3.5)],
        EVENT_REFERENCE_SCORE: [(36, 1.0), (60, 2.5)],
        EVENT_ID: [(36, 1), (60, 2)],
        ACCRUAL_EVENT_ID: [(36, 1), (60, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_1,
        relapse_timestamps=[6, 30],
        targets_dict=test_case_1_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 10,
            "opt_raw_before_relapse_max_time": 4,
            "opt_raw_after_relapse_max_time": 4,
        },
    ), "Test 1 failed!"
    # Test case 2 - with relapse, unconfirmed
    test_dataframe_case_2 = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
        }
    )
    test_case_2_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 55),
            (10, 45),
            (20, 35),
            (30, 25),
            (40, 15),
            (50, 5),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (60, 5),
            (70, 15),
            (80, 25),
            (90, 35),
            (100, 45),
            (110, 55),
            (120, 65),
        ],
        IS_POST_EVENT_REBASELINE: [(30, True), (60, True), (100, True)],
        IS_PIRA_REBASELINE: [(30, True), (60, True), (80, True), (100, True)],
        IS_GENERAL_REBASELINE: [(30, True), (60, True), (100, True)],
        IS_POST_RELAPSE_REBASELINE: [(80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 2.5),
            (60, 4.0),
            (80, 4.5),
            (100, 6.5),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.5), (60, 4.0), (100, 6.5)],
        IS_EVENT: [(30, True), (60, True), (100, True)],
        IS_ACCRUAL_EVENT: [(30, True), (60, True), (100, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (60, LABEL_RAW), (100, LABEL_PIRA)],
        EVENT_SCORE: [(30, 2.5), (60, 4.0), (100, 6.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (60, 2.5), (100, 4.5)],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (100, 3),
            (110, 3),
            (120, 3),
        ],
        ACCRUAL_EVENT_ID: [
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
        follow_up_dataframe=test_dataframe_case_2,
        relapse_timestamps=[55],
        targets_dict=test_case_2_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "re-baselining only",
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 15,
        },
    ), "Test 2 failed!"

    # Test case 3 - with relapse, next-confirmed
    test_case_3_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 55),
            (10, 45),
            (20, 35),
            (30, 25),
            (40, 15),
            (50, 5),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (60, 5),
            (70, 15),
            (80, 25),
            (90, 35),
            (100, 45),
            (110, 55),
            (120, 65),
        ],
        IS_POST_EVENT_REBASELINE: [(30, True), (50, True), (100, True)],
        IS_PIRA_REBASELINE: [(30, True), (50, True), (80, True), (100, True)],
        IS_GENERAL_REBASELINE: [(30, True), (50, True), (100, True)],
        IS_POST_RELAPSE_REBASELINE: [(80, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 2.0),
            (50, 4.0),
            (80, 4.5),
            (100, 6.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.0), (50, 4.0), (100, 6.0)],
        IS_EVENT: [(30, True), (50, True), (100, True)],
        IS_ACCRUAL_EVENT: [(30, True), (50, True), (100, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (50, LABEL_RAW), (100, LABEL_PIRA)],
        EVENT_SCORE: [(30, 2.0), (50, 4.0), (100, 6.0)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (50, 2.0), (100, 4.5)],
        EVENT_ID: [
            (30, 1),
            (50, 2),
            (60, 2),
            (70, 2),
            (100, 3),
            (110, 3),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (50, 2),
            (60, 2),
            (70, 2),
            (100, 3),
            (110, 3),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_2,
        relapse_timestamps=[55],
        targets_dict=test_case_3_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "re-baselining only",
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 15,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 3 failed!"

    # Test case 4 - with relapse, unconfirmed
    test_dataframe_case_4 = pd.DataFrame(
        {
            TIMESTAMP: [
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
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
        }
    )
    test_case_4_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 65),
            (10, 55),
            (20, 45),
            (30, 35),
            (40, 25),
            (50, 15),
            (60, 5),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (70, 5),
            (80, 15),
            (90, 25),
            (100, 35),
            (110, 45),
            (120, 55),
        ],
        IS_POST_EVENT_REBASELINE: [(30, True), (70, True), (110, True)],
        IS_PIRA_REBASELINE: [(30, True), (70, True), (90, True), (110, True)],
        IS_GENERAL_REBASELINE: [(30, True), (70, True), (110, True)],
        IS_POST_RELAPSE_REBASELINE: [(90, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 3.0),
            (70, 4.5),
            (90, 5.0),
            (110, 6.5),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.0), (70, 4.5), (110, 6.5)],
        IS_EVENT: [(30, True), (70, True), (110, True)],
        IS_ACCRUAL_EVENT: [(30, True), (70, True), (110, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (70, LABEL_RAW), (110, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.0), (70, 4.5), (110, 6.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (70, 3.0), (110, 5.0)],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (70, 2),
            (80, 2),
            (110, 3),
            (120, 3),
        ],
        ACCRUAL_EVENT_ID: [
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
        follow_up_dataframe=test_dataframe_case_4,
        relapse_timestamps=[65],
        targets_dict=test_case_4_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "re-baselining only",
            "opt_require_confirmation": False,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 15,
        },
    ), "Test 4 failed!"

    # Test case 5 - with relapse, unconfirmed
    test_case_5_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 65),
            (10, 55),
            (20, 45),
            (30, 35),
            (40, 25),
            (50, 15),
            (60, 5),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (70, 5),
            (80, 15),
            (90, 25),
            (100, 35),
            (110, 45),
            (120, 55),
        ],
        IS_POST_EVENT_REBASELINE: [(30, True), (60, True), (110, True)],
        IS_PIRA_REBASELINE: [(30, True), (60, True), (90, True), (110, True)],
        IS_GENERAL_REBASELINE: [(30, True), (60, True), (110, True)],
        IS_POST_RELAPSE_REBASELINE: [(90, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 2.5),
            (60, 4.5),
            (90, 5.0),
            (110, 6.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 2.5), (60, 4.5), (110, 6.0)],
        IS_EVENT: [(30, True), (60, True), (110, True)],
        IS_ACCRUAL_EVENT: [(30, True), (60, True), (110, True)],
        EVENT_TYPE: [(30, LABEL_PIRA), (60, LABEL_RAW), (110, LABEL_PIRA)],
        EVENT_SCORE: [(30, 2.5), (60, 4.5), (110, 6.0)],
        EVENT_REFERENCE_SCORE: [(30, 1.0), (60, 2.5), (110, 5.0)],
        EVENT_ID: [(30, 1), (40, 1), (60, 2), (70, 2), (80, 2), (110, 3)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (60, 2), (70, 2), (80, 2), (110, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_4,
        relapse_timestamps=[65],
        targets_dict=test_case_5_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "re-baselining only",
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 15,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 5 failed!"

    # Test case 6 - stagnation then increase, unconfirmed
    test_dataframe_case_6 = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 40, 50, 60, 65, 70],
            EDSS_SCORE: [1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0],
        }
    )
    test_case_6_targets = {
        DAYS_TO_NEXT_RELAPSE: [],
        DAYS_SINCE_PREVIOUS_RELAPSE: [],
        IS_POST_EVENT_REBASELINE: [(30, True)],
        IS_PIRA_REBASELINE: [(30, True)],
        IS_GENERAL_REBASELINE: [(30, True)],
        IS_POST_RELAPSE_REBASELINE: [],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 4.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 4.0)],
        IS_EVENT: [(30, True)],
        IS_ACCRUAL_EVENT: [(30, True)],
        EVENT_TYPE: [(30, LABEL_PIRA)],
        EVENT_SCORE: [(30, 4.0)],
        EVENT_REFERENCE_SCORE: [(30, 1.0)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (65, 1), (70, 1)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1), (65, 1), (70, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_6,
        relapse_timestamps=[],
        targets_dict=test_case_6_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": False,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 6 failed!"

    # Test case 7 - stagnation then increase, next-confirmed
    test_case_7_targets = {
        DAYS_TO_NEXT_RELAPSE: [],
        DAYS_SINCE_PREVIOUS_RELAPSE: [],
        IS_POST_EVENT_REBASELINE: [(30, True)],
        IS_PIRA_REBASELINE: [(30, True)],
        IS_GENERAL_REBASELINE: [(30, True)],
        IS_POST_RELAPSE_REBASELINE: [],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 3.5),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 3.5)],
        IS_EVENT: [(30, True)],
        IS_ACCRUAL_EVENT: [(30, True)],
        EVENT_TYPE: [(30, LABEL_PIRA)],
        EVENT_SCORE: [(30, 3.5)],
        EVENT_REFERENCE_SCORE: [(30, 1.0)],
        EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1)],
        ACCRUAL_EVENT_ID: [(30, 1), (40, 1), (50, 1), (60, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_6,
        relapse_timestamps=[],
        targets_dict=test_case_7_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 0.5,
        },
    ), "Test 7 failed!"

    # Test case 8 - maximal distance settings
    test_dataframe_case_8 = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 20, 30, 50, 60, 65, 70],  # , 40
            EDSS_SCORE: [1, 1, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5],  # , 2.5
        }
    )
    test_case_8_targets = {
        DAYS_TO_NEXT_RELAPSE: [],
        DAYS_SINCE_PREVIOUS_RELAPSE: [],
        IS_POST_EVENT_REBASELINE: [(30, True), (50, True), (65, True)],
        IS_PIRA_REBASELINE: [(30, True), (50, True), (65, True)],
        IS_GENERAL_REBASELINE: [(30, True), (50, True), (65, True)],
        IS_POST_RELAPSE_REBASELINE: [],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 2.0),
            (50, 3.0),
            (65, 4.5),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 2.0),
            (50, 3.0),
            (65, 4.5),
        ],
        IS_EVENT: [(30, True), (50, True), (65, True)],
        IS_ACCRUAL_EVENT: [(30, True), (50, True), (65, True)],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (50, LABEL_PIRA),
            (65, LABEL_PIRA),
        ],
        EVENT_SCORE: [
            (30, 2.0),
            (50, 3.0),
            (65, 4.5),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (50, 2.0),
            (65, 3.0),
        ],
        EVENT_ID: [(30, 1), (50, 2), (65, 3), (70, 3)],
        ACCRUAL_EVENT_ID: [(30, 1), (50, 2), (65, 3), (70, 3)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_8,
        relapse_timestamps=[],
        targets_dict=test_case_8_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "continuous_events_max_merge_distance": 9.9,
            "opt_require_confirmation": False,
        },
    ), "Test 8 failed!"

    # Test case 9 - maximal distance settings
    test_case_9_targets = {
        DAYS_TO_NEXT_RELAPSE: [],
        DAYS_SINCE_PREVIOUS_RELAPSE: [],
        IS_POST_EVENT_REBASELINE: [(30, True), (50, True)],
        IS_PIRA_REBASELINE: [(30, True), (50, True)],
        IS_GENERAL_REBASELINE: [(30, True), (50, True)],
        IS_POST_RELAPSE_REBASELINE: [],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 2.0),
            (50, 4.5),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 2.0),
            (50, 4.5),
        ],
        IS_EVENT: [(30, True), (50, True)],
        IS_ACCRUAL_EVENT: [(30, True), (50, True)],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (50, LABEL_PIRA),
        ],
        EVENT_SCORE: [
            (30, 2.0),
            (50, 4.5),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (50, 2.0),
        ],
        EVENT_ID: [(30, 1), (50, 2), (60, 2), (65, 2), (70, 2)],
        ACCRUAL_EVENT_ID: [(30, 1), (50, 2), (60, 2), (65, 2), (70, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_8,
        relapse_timestamps=[],
        targets_dict=test_case_9_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "continuous_events_max_merge_distance": 10,
            "opt_require_confirmation": False,
        },
    ), "Test 9 failed!"

    # Test case 10 - maximal distance settings
    test_case_10_targets = {
        DAYS_TO_NEXT_RELAPSE: [],
        DAYS_SINCE_PREVIOUS_RELAPSE: [],
        IS_POST_EVENT_REBASELINE: [(30, True)],
        IS_PIRA_REBASELINE: [(30, True)],
        IS_GENERAL_REBASELINE: [(30, True)],
        IS_POST_RELAPSE_REBASELINE: [],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 4.5),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 4.5),
        ],
        IS_EVENT: [(30, True)],
        IS_ACCRUAL_EVENT: [(30, True)],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
        ],
        EVENT_SCORE: [
            (30, 4.5),
        ],
        EVENT_REFERENCE_SCORE: [(30, 1.0)],
        EVENT_ID: [(30, 1), (50, 1), (60, 1), (65, 1), (70, 1)],
        ACCRUAL_EVENT_ID: [(30, 1), (50, 1), (60, 1), (65, 1), (70, 1)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_8,
        relapse_timestamps=[],
        targets_dict=test_case_10_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 10 failed!"

    ### Part 2 - new features and fixes ###
    # Different event types are never merged
    test_dataframe_case_11 = pd.DataFrame(
        {
            TIMESTAMP: [
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
                130,
                140,
                150,
            ],
            EDSS_SCORE: [
                1,
                1,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
                5.0,
                5.5,
                6.0,
                6.5,
                7.0,
                7.5,
                8.0,
            ],
        }
    )
    test_case_11_ff_common_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 85),
            (10, 75),
            (20, 65),
            (30, 55),
            (40, 45),
            (50, 35),
            (60, 25),
            (70, 15),
            (80, 5),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (90, 5),
            (100, 15),
            (110, 25),
            (120, 35),
            (130, 45),
            (140, 55),
            (150, 65),
        ],
        IS_POST_RELAPSE_REBASELINE: [(110, True)],
    }
    test_case_11_targets = {
        IS_POST_EVENT_REBASELINE: [(30, True), (90, True), (110, True), (120, True)],
        IS_PIRA_REBASELINE: [(30, True), (90, True), (110, True), (120, True)],
        IS_GENERAL_REBASELINE: [(30, True), (90, True), (110, True), (120, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 4.0),
            (90, 5.5),
            (110, 6.0),
            (120, 8.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 4.0),
            (90, 5.5),
            (110, 6.0),
            (120, 8.0),
        ],
        IS_EVENT: [(30, True), (90, True), (110, True), (120, True)],
        IS_ACCRUAL_EVENT: [(30, True), (90, True), (110, True), (120, True)],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (90, LABEL_RAW),
            (110, LABEL_UNDEFINED),
            (120, LABEL_PIRA),
        ],
        EVENT_SCORE: [
            (30, 4.0),
            (90, 5.5),
            (110, 6.0),
            (120, 8.0),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (90, 4.0),
            (110, 5.5),
            (120, 6.0),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (90, 2),
            (100, 2),
            (110, 3),
            (120, 4),
            (130, 4),
            (140, 4),
            (150, 4),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (90, 2),
            (100, 2),
            (110, 3),
            (120, 4),
            (130, 4),
            (140, 4),
            (150, 4),
        ],
    }
    test_case_11_targets = {**test_case_11_targets, **test_case_11_ff_common_targets}
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_11,
        relapse_timestamps=[85],
        targets_dict=test_case_11_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 20,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": False,
        },
    ), "Test 11 failed!"
    test_case_12_targets = {
        IS_POST_EVENT_REBASELINE: [
            (30, True),
            (70, True),
            (90, True),
            (110, True),
            (120, True),
        ],
        IS_PIRA_REBASELINE: [
            (30, True),
            (70, True),
            (90, True),
            (110, True),
            (120, True),
        ],
        IS_GENERAL_REBASELINE: [
            (30, True),
            (70, True),
            (90, True),
            (110, True),
            (120, True),
        ],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 3.0),
            (70, 4.0),
            (90, 5.5),
            (110, 6.0),
            (120, 7.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 3.0),
            (70, 4.0),
            (90, 5.5),
            (110, 6.0),
            (120, 7.0),
        ],
        IS_EVENT: [(30, True), (70, True), (90, True), (110, True), (120, True)],
        IS_ACCRUAL_EVENT: [
            (30, True),
            (70, True),
            (90, True),
            (110, True),
            (120, True),
        ],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (70, LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW),
            (90, LABEL_RAW),
            (110, LABEL_UNDEFINED),
            (120, LABEL_PIRA),
        ],
        EVENT_SCORE: [
            (30, 3.0),
            (70, 4.0),
            (90, 5.5),
            (110, 6.0),
            (120, 7.0),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (70, 3.0),
            (90, 4.0),
            (110, 5.5),
            (120, 6.0),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (70, 2),
            (90, 3),
            (100, 3),
            (110, 4),
            (120, 5),
            (130, 5),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (70, 2),
            (90, 3),
            (100, 3),
            (110, 4),
            (120, 5),
            (130, 5),
        ],
    }
    test_case_12_targets = {**test_case_12_targets, **test_case_11_ff_common_targets}
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_11,
        relapse_timestamps=[85],
        targets_dict=test_case_12_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 20,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
        },
    ), "Test 12 failed!"
    test_case_13_targets = {
        IS_POST_EVENT_REBASELINE: [
            (30, True),
            (70, True),
            (90, True),
            (120, True),
        ],
        IS_PIRA_REBASELINE: [
            (30, True),
            (70, True),
            (90, True),
            (110, True),
            (120, True),
        ],
        IS_GENERAL_REBASELINE: [
            (30, True),
            (70, True),
            (90, True),
            (120, True),
        ],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 3.0),
            (70, 4.0),
            (90, 5.5),
            (110, 6.0),
            (120, 7.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 3.0),
            (70, 4.0),
            (90, 5.5),
            (120, 7.0),
        ],
        IS_EVENT: [(30, True), (70, True), (90, True), (120, True)],
        IS_ACCRUAL_EVENT: [
            (30, True),
            (70, True),
            (90, True),
            (120, True),
        ],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (70, LABEL_PIRA_CONFIRMED_IN_RAW_WINDOW),
            (90, LABEL_RAW),
            (120, LABEL_PIRA),
        ],
        EVENT_SCORE: [
            (30, 3.0),
            (70, 4.0),
            (90, 5.5),
            (120, 7.0),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (70, 3.0),
            (90, 4.0),
            (120, 6.0),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (70, 2),
            (90, 3),
            (100, 3),
            (120, 4),
            (130, 4),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (70, 2),
            (90, 3),
            (100, 3),
            (120, 4),
            (130, 4),
        ],
    }
    test_case_13_targets = {**test_case_13_targets, **test_case_11_ff_common_targets}
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_11,
        relapse_timestamps=[85],
        targets_dict=test_case_13_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "never",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 20,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": 5,
            "opt_require_confirmation": True,
            "opt_confirmation_time": 20,
        },
    ), "Test 13 failed!"
    # Relapses stop the merging of PIRA events
    test_dataframe_case_14 = pd.DataFrame(
        {
            TIMESTAMP: [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                110,
                120,
                130,
                140,
                150,
            ],
            EDSS_SCORE: [
                1,
                1,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.0,
                6.5,
                7.0,
                7.5,
                8.0,
            ],
        }
    )
    test_case_14_targets = {
        DAYS_TO_NEXT_RELAPSE: [],
        DAYS_SINCE_PREVIOUS_RELAPSE: [],
        IS_POST_RELAPSE_REBASELINE: [],
        IS_POST_EVENT_REBASELINE: [
            (30, True),
        ],
        IS_PIRA_REBASELINE: [
            (30, True),
        ],
        IS_GENERAL_REBASELINE: [
            (30, True),
        ],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 8.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 8.0),
        ],
        IS_EVENT: [(30, True)],
        IS_ACCRUAL_EVENT: [
            (30, True),
        ],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
        ],
        EVENT_SCORE: [
            (30, 8.0),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (110, 1),
            (120, 1),
            (130, 1),
            (140, 1),
            (150, 1),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (110, 1),
            (120, 1),
            (130, 1),
            (140, 1),
            (150, 1),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_14,
        relapse_timestamps=[],
        targets_dict=test_case_14_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 20,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 14 failed!"
    test_case_15_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 85),
            (10, 75),
            (20, 65),
            (30, 55),
            (40, 45),
            (50, 35),
            (60, 25),
            (70, 15),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (110, 25),
            (120, 35),
            (130, 45),
            (140, 55),
            (150, 65),
        ],
        IS_POST_EVENT_REBASELINE: [(30, True), (120, True)],
        IS_PIRA_REBASELINE: [(30, True), (120, True)],
        IS_GENERAL_REBASELINE: [(30, True), (120, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 4.0), (120, 8.0)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 4.0), (120, 8.0)],
        IS_EVENT: [(30, True), (120, True)],
        IS_ACCRUAL_EVENT: [(30, True), (120, True)],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (120, LABEL_PIRA),
        ],
        EVENT_SCORE: [(30, 4.0), (120, 8.0)],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (120, 4.0),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (120, 2),
            (130, 2),
            (140, 2),
            (150, 2),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (120, 2),
            (130, 2),
            (140, 2),
            (150, 2),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_14,
        relapse_timestamps=[85],
        targets_dict=test_case_15_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 20,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 15 failed!"
    # This does not rely on undefined
    test_dataframe_case_16 = pd.DataFrame(
        {
            TIMESTAMP: [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                110,
                120,
                130,
                140,
                150,
            ],
            EDSS_SCORE: [
                1,
                1,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                6.0,
                6.5,
                7.0,
                7.5,
                8.0,
            ],
        }
    )
    test_case_16_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 85),
            (10, 75),
            (20, 65),
            (30, 55),
            (40, 45),
            (50, 35),
            (60, 25),
            (70, 15),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (110, 25),
            (120, 35),
            (130, 45),
            (140, 55),
            (150, 65),
        ],
        IS_POST_RELAPSE_REBASELINE: [(110, True)],
        IS_POST_EVENT_REBASELINE: [(30, True), (110, True), (120, True)],
        IS_PIRA_REBASELINE: [(30, True), (110, True), (120, True)],
        IS_GENERAL_REBASELINE: [(30, True), (110, True), (120, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 4.0), (110, 6.0), (120, 8.0)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 4.0), (110, 6.0), (120, 8.0)],
        IS_EVENT: [(30, True), (110, True), (120, True)],
        IS_ACCRUAL_EVENT: [(30, True), (110, True), (120, True)],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (110, LABEL_UNDEFINED),
            (120, LABEL_PIRA),
        ],
        EVENT_SCORE: [(30, 4.0), (110, 6.0), (120, 8.0)],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (110, 4.0),
            (120, 6.0),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (110, 2),
            (120, 3),
            (130, 3),
            (140, 3),
            (150, 3),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (110, 2),
            (120, 3),
            (130, 3),
            (140, 3),
            (150, 3),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_16,
        relapse_timestamps=[85],
        targets_dict=test_case_16_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 20,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 16 failed!"
    test_case_17_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 85),
            (10, 75),
            (20, 65),
            (30, 55),
            (40, 45),
            (50, 35),
            (60, 25),
            (70, 15),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (110, 25),
            (120, 35),
            (130, 45),
            (140, 55),
            (150, 65),
        ],
        IS_POST_RELAPSE_REBASELINE: [(110, True)],
        IS_POST_EVENT_REBASELINE: [(30, True), (120, True)],
        IS_PIRA_REBASELINE: [(30, True), (110, True), (120, True)],
        IS_GENERAL_REBASELINE: [(30, True), (120, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(30, 4.0), (110, 6.0), (120, 8.0)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(30, 4.0), (120, 8.0)],
        IS_EVENT: [(30, True), (120, True)],
        IS_ACCRUAL_EVENT: [(30, True), (120, True)],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (120, LABEL_PIRA),
        ],
        EVENT_SCORE: [(30, 4.0), (120, 8.0)],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (120, 6.0),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (120, 2),
            (130, 2),
            (140, 2),
            (150, 2),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (120, 2),
            (130, 2),
            (140, 2),
            (150, 2),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_16,
        relapse_timestamps=[85],
        targets_dict=test_case_17_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "never",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 20,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 17 failed!"
    # Relapse-free periods stop the merging of RAW
    test_case_18_ff_common_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 55),
            (10, 45),
            (20, 35),
            (30, 25),
            (40, 15),
            (50, 5),
            (60, 55),
            (70, 45),
            (110, 5),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (60, 5),
            (70, 15),
            (110, 55),
            (120, 5),
            (130, 15),
            (140, 25),
            (150, 35),
        ],
        IS_POST_RELAPSE_REBASELINE: [(110, True), (140, True)],
    }
    test_case_18_targets = {
        IS_POST_EVENT_REBASELINE: [
            (30, True),
            (60, True),
            (110, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        IS_PIRA_REBASELINE: [
            (30, True),
            (60, True),
            (110, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        IS_GENERAL_REBASELINE: [
            (30, True),
            (60, True),
            (110, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 2.5),
            (60, 4.0),
            (110, 6.0),
            (120, 7.0),
            (140, 7.5),
            (150, 8.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 2.5),
            (60, 4.0),
            (110, 6.0),
            (120, 7.0),
            (140, 7.5),
            (150, 8.0),
        ],
        IS_EVENT: [
            (30, True),
            (60, True),
            (110, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        IS_ACCRUAL_EVENT: [
            (30, True),
            (60, True),
            (110, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (60, LABEL_RAW),
            (110, LABEL_UNDEFINED),
            (120, LABEL_RAW),
            (140, LABEL_UNDEFINED),
            (150, LABEL_PIRA),
        ],
        EVENT_SCORE: [
            (30, 2.5),
            (60, 4.0),
            (110, 6.0),
            (120, 7.0),
            (140, 7.5),
            (150, 8.0),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (60, 2.5),
            (110, 4.0),
            (120, 6.0),
            (140, 7.0),
            (150, 7.5),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (110, 3),
            (120, 4),
            (130, 4),
            (140, 5),
            (150, 6),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (110, 3),
            (120, 4),
            (130, 4),
            (140, 5),
            (150, 6),
        ],
    }
    test_case_18_targets = {**test_case_18_targets, **test_case_18_ff_common_targets}
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_16,
        relapse_timestamps=[55, 115],
        targets_dict=test_case_18_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 15,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 18 failed!"
    test_case_19_targets = {
        IS_POST_EVENT_REBASELINE: [
            (30, True),
            (60, True),
            (120, True),
            (150, True),
        ],
        IS_PIRA_REBASELINE: [
            (30, True),
            (60, True),
            (110, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        IS_GENERAL_REBASELINE: [
            (30, True),
            (60, True),
            (120, True),
            (150, True),
        ],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 2.5),
            (60, 4.0),
            (110, 6.0),
            (120, 7.0),
            (140, 7.5),
            (150, 8.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 2.5),
            (60, 4.0),
            (120, 7.0),
            (150, 8.0),
        ],
        IS_EVENT: [
            (30, True),
            (60, True),
            (120, True),
            (150, True),
        ],
        IS_ACCRUAL_EVENT: [
            (30, True),
            (60, True),
            (120, True),
            (150, True),
        ],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (60, LABEL_RAW),
            (120, LABEL_RAW),
            (150, LABEL_PIRA),
        ],
        EVENT_SCORE: [
            (30, 2.5),
            (60, 4.0),
            (120, 7.0),
            (150, 8.0),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (60, 2.5),
            (120, 4.0),
            (150, 7.5),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (120, 3),
            (130, 3),
            (150, 4),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (120, 3),
            (130, 3),
            (150, 4),
        ],
    }
    test_case_19_targets = {**test_case_19_targets, **test_case_18_ff_common_targets}
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_16,
        relapse_timestamps=[55, 115],
        targets_dict=test_case_19_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "never",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 15,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 19 failed!"
    # 'Relapse-free' depends on the RAW window size; events
    # are merged if windows overlap.
    test_case_20_targets = {
        IS_POST_EVENT_REBASELINE: [
            (30, True),
            (60, True),
        ],
        IS_PIRA_REBASELINE: [
            (30, True),
            (60, True),
        ],
        IS_GENERAL_REBASELINE: [
            (30, True),
            (60, True),
        ],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 2.5),
            (60, 8.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 2.5),
            (60, 8.0),
        ],
        IS_EVENT: [
            (30, True),
            (60, True),
        ],
        IS_ACCRUAL_EVENT: [
            (30, True),
            (60, True),
        ],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (60, LABEL_RAW),
        ],
        EVENT_SCORE: [
            (30, 2.5),
            (60, 8.0),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (60, 2.5),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (110, 2),
            (120, 2),
            (130, 2),
            (140, 2),
            (150, 2),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (110, 2),
            (120, 2),
            (130, 2),
            (140, 2),
            (150, 2),
        ],
    }
    test_case_20_targets = {**test_case_20_targets, **test_case_18_ff_common_targets}
    test_case_20_targets[IS_POST_RELAPSE_REBASELINE] = []
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_16,
        relapse_timestamps=[55, 115],
        targets_dict=test_case_20_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "never",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 65,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 20 failed!"
    test_dataframe_case_21 = test_dataframe_case_16.copy()
    test_dataframe_case_21.at[8, EDSS_SCORE] = 4.0
    test_case_21_targets = {
        IS_POST_EVENT_REBASELINE: [
            (30, True),
            (60, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        IS_PIRA_REBASELINE: [
            (30, True),
            (60, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        IS_GENERAL_REBASELINE: [
            (30, True),
            (60, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 2.5),
            (60, 4.0),
            (120, 7.0),
            (140, 7.5),
            (150, 8.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 2.5),
            (60, 4.0),
            (120, 7.0),
            (140, 7.5),
            (150, 8.0),
        ],
        IS_EVENT: [
            (30, True),
            (60, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        IS_ACCRUAL_EVENT: [
            (30, True),
            (60, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (60, LABEL_RAW),
            (120, LABEL_RAW),
            (140, LABEL_UNDEFINED),
            (150, LABEL_PIRA),
        ],
        EVENT_SCORE: [
            (30, 2.5),
            (60, 4.0),
            (120, 7.0),
            (140, 7.5),
            (150, 8.0),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (60, 2.5),
            (120, 4.0),
            (140, 7.0),
            (150, 7.5),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (120, 3),
            (130, 3),
            (140, 4),
            (150, 5),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (120, 3),
            (130, 3),
            (140, 4),
            (150, 5),
        ],
    }
    test_case_21_targets = {**test_case_21_targets, **test_case_18_ff_common_targets}
    test_case_21_targets[IS_POST_RELAPSE_REBASELINE] = [(140, True)]
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_21,
        relapse_timestamps=[55, 115],
        targets_dict=test_case_21_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 15,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 21 failed!"
    test_case_22_targets = {
        IS_POST_EVENT_REBASELINE: [
            (30, True),
            (60, True),
            (120, True),
            (150, True),
        ],
        IS_PIRA_REBASELINE: [
            (30, True),
            (60, True),
            (120, True),
            (140, True),
            (150, True),
        ],
        IS_GENERAL_REBASELINE: [
            (30, True),
            (60, True),
            (120, True),
            (150, True),
        ],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 2.5),
            (60, 4.0),
            (120, 7.0),
            (140, 7.5),
            (150, 8.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 2.5),
            (60, 4.0),
            (120, 7.0),
            (150, 8.0),
        ],
        IS_EVENT: [
            (30, True),
            (60, True),
            (120, True),
            (150, True),
        ],
        IS_ACCRUAL_EVENT: [
            (30, True),
            (60, True),
            (120, True),
            (150, True),
        ],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (60, LABEL_RAW),
            (120, LABEL_RAW),
            (150, LABEL_PIRA),
        ],
        EVENT_SCORE: [
            (30, 2.5),
            (60, 4.0),
            (120, 7.0),
            (150, 8.0),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (60, 2.5),
            (120, 4.0),
            (150, 7.5),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (120, 3),
            (130, 3),
            (150, 4),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (120, 3),
            (130, 3),
            (150, 4),
        ],
    }
    test_case_22_targets = {**test_case_22_targets, **test_case_18_ff_common_targets}
    test_case_22_targets[IS_POST_RELAPSE_REBASELINE] = [(140, True)]
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_21,
        relapse_timestamps=[55, 115],
        targets_dict=test_case_22_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "never",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 15,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 22 failed!"
    test_case_23_targets = {
        IS_POST_EVENT_REBASELINE: [
            (30, True),
            (60, True),
        ],
        IS_PIRA_REBASELINE: [
            (30, True),
            (60, True),
        ],
        IS_GENERAL_REBASELINE: [
            (30, True),
            (60, True),
        ],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [
            (30, 2.5),
            (60, 8.0),
        ],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [
            (30, 2.5),
            (60, 8.0),
        ],
        IS_EVENT: [
            (30, True),
            (60, True),
        ],
        IS_ACCRUAL_EVENT: [
            (30, True),
            (60, True),
        ],
        EVENT_TYPE: [
            (30, LABEL_PIRA),
            (60, LABEL_RAW),
        ],
        EVENT_SCORE: [
            (30, 2.5),
            (60, 8.0),
        ],
        EVENT_REFERENCE_SCORE: [
            (30, 1.0),
            (60, 2.5),
        ],
        EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (110, 2),
            (120, 2),
            (130, 2),
            (140, 2),
            (150, 2),
        ],
        ACCRUAL_EVENT_ID: [
            (30, 1),
            (40, 1),
            (60, 2),
            (70, 2),
            (110, 2),
            (120, 2),
            (130, 2),
            (140, 2),
            (150, 2),
        ],
    }
    test_case_23_targets = {**test_case_23_targets, **test_case_18_ff_common_targets}
    test_case_23_targets[IS_POST_RELAPSE_REBASELINE] = []
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_21,
        relapse_timestamps=[55, 115],
        targets_dict=test_case_23_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 65,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 23 failed!"
    # Relapses don't affect merging of undefined events
    test_dataframe_case_24 = test_dataframe_case_16.copy()
    test_dataframe_case_24[EDSS_SCORE] = 9 - test_dataframe_case_24[EDSS_SCORE]
    test_case_24_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 85),
            (10, 75),
            (20, 65),
            (30, 55),
            (40, 45),
            (50, 35),
            (60, 25),
            (70, 15),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (110, 25),
            (120, 35),
            (130, 45),
            (140, 55),
            (150, 65),
        ],
        IS_POST_RELAPSE_REBASELINE: [],
        IS_POST_EVENT_REBASELINE: [(20, True)],
        IS_PIRA_REBASELINE: [(20, True)],
        IS_GENERAL_REBASELINE: [(20, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 1.0)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 1.0)],
        IS_EVENT: [(20, True)],
        IS_IMPROVEMENT_EVENT: [(20, True)],
        EVENT_TYPE: [
            (20, LABEL_IMPROVEMENT),
        ],
        EVENT_SCORE: [(20, 1.0)],
        EVENT_REFERENCE_SCORE: [
            (20, 8.0),
        ],
        EVENT_ID: [
            (20, 1),
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (110, 1),
            (120, 1),
            (130, 1),
            (140, 1),
            (150, 1),
        ],
        IMPROVEMENT_EVENT_ID: [
            (20, 1),
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (110, 1),
            (120, 1),
            (130, 1),
            (140, 1),
            (150, 1),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_24,
        relapse_timestamps=[85],
        targets_dict=test_case_24_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 20,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 24 failed!"
    test_case_25_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 55),
            (10, 45),
            (20, 35),
            (30, 25),
            (40, 15),
            (50, 5),
            (60, 55),
            (70, 45),
            (110, 5),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (60, 5),
            (70, 15),
            (110, 55),
            (120, 5),
            (130, 15),
            (140, 25),
            (150, 35),
        ],
        IS_POST_RELAPSE_REBASELINE: [],
        IS_POST_EVENT_REBASELINE: [(20, True)],
        IS_PIRA_REBASELINE: [(20, True)],
        IS_GENERAL_REBASELINE: [(20, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 1.0)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 1.0)],
        IS_EVENT: [(20, True)],
        IS_IMPROVEMENT_EVENT: [(20, True)],
        EVENT_TYPE: [
            (20, LABEL_IMPROVEMENT),
        ],
        EVENT_SCORE: [(20, 1.0)],
        EVENT_REFERENCE_SCORE: [
            (20, 8.0),
        ],
        EVENT_ID: [
            (20, 1),
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (110, 1),
            (120, 1),
            (130, 1),
            (140, 1),
            (150, 1),
        ],
        IMPROVEMENT_EVENT_ID: [
            (20, 1),
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (110, 1),
            (120, 1),
            (130, 1),
            (140, 1),
            (150, 1),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_24,
        relapse_timestamps=[55, 115],
        targets_dict=test_case_25_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 20,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 25 failed!"
    test_case_26_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 45),
            (10, 35),
            (20, 25),
            (30, 15),
            (40, 5),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (50, 5),
            (60, 15),
            (70, 25),
            (110, 65),
            (120, 75),
            (130, 85),
            (140, 95),
            (150, 105),
        ],
        IS_POST_RELAPSE_REBASELINE: [],
        IS_POST_EVENT_REBASELINE: [(20, True)],
        IS_PIRA_REBASELINE: [(20, True)],
        IS_GENERAL_REBASELINE: [(20, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(20, 1.0)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(20, 1.0)],
        IS_EVENT: [(20, True)],
        IS_IMPROVEMENT_EVENT: [(20, True)],
        EVENT_TYPE: [
            (20, LABEL_IMPROVEMENT),
        ],
        EVENT_SCORE: [(20, 1.0)],
        EVENT_REFERENCE_SCORE: [
            (20, 8.0),
        ],
        EVENT_ID: [
            (20, 1),
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (110, 1),
            (120, 1),
            (130, 1),
            (140, 1),
            (150, 1),
        ],
        IMPROVEMENT_EVENT_ID: [
            (20, 1),
            (30, 1),
            (40, 1),
            (50, 1),
            (60, 1),
            (70, 1),
            (110, 1),
            (120, 1),
            (130, 1),
            (140, 1),
            (150, 1),
        ],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_24,
        relapse_timestamps=[45],
        targets_dict=test_case_26_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 20,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 26 failed!"
    # Undefined events are not merged
    test_dataframe_case_27 = pd.DataFrame(
        {
            TIMESTAMP: [0, 10, 40, 50, 60, 70],
            EDSS_SCORE: [1, 1.5, 2.5, 3, 3.5, 4.0],
        }
    )
    test_case_27_targets = {
        DAYS_TO_NEXT_RELAPSE: [
            (0, 5),
            (10, 15),
        ],
        DAYS_SINCE_PREVIOUS_RELAPSE: [
            (10, 5),
            (40, 15),
            (50, 25),
            (60, 35),
            (70, 45),
        ],
        IS_POST_RELAPSE_REBASELINE: [(40, True)],
        IS_POST_EVENT_REBASELINE: [(40, True), (60, True)],
        IS_PIRA_REBASELINE: [(40, True), (60, True)],
        IS_GENERAL_REBASELINE: [(40, True), (60, True)],
        EDSS_SCORE_USED_AS_NEW_PIRA_REFERENCE: [(40, 2.5), (60, 4.0)],
        EDSS_SCORE_USED_AS_NEW_GENERAL_REFERENCE: [(40, 2.5), (60, 4.0)],
        IS_EVENT: [(40, True), (60, True)],
        IS_ACCRUAL_EVENT: [(40, True), (60, True)],
        EVENT_TYPE: [(40, LABEL_UNDEFINED), (60, LABEL_PIRA)],
        EVENT_SCORE: [(40, 2.5), (60, 4.0)],
        EVENT_REFERENCE_SCORE: [(40, 1.0), (60, 2.5)],
        EVENT_ID: [(40, 1), (60, 2), (70, 2)],
        ACCRUAL_EVENT_ID: [(40, 1), (60, 2), (70, 2)],
    }
    assert raw_pira_progression_result_is_equal_to_target(
        follow_up_dataframe=test_dataframe_case_27,
        relapse_timestamps=[5, 25],
        targets_dict=test_case_27_targets,
        args_dict={
            "annotation_mode": SYMMETRIC_MODE_NAME,
            "opt_baseline_type": "fixed",
            "undefined_events_annotation_mode": "all",
            "opt_raw_before_relapse_max_time": 5,
            "opt_raw_after_relapse_max_time": 10,
            "merge_continuous_events": True,
            "continuous_events_max_repetition_time": np.inf,
            "continuous_events_max_merge_distance": np.inf,
            "opt_require_confirmation": False,
        },
    ), "Test 27 failed!"


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

    print("Testing '_check_assessment_for_event'...")
    test_check_assessment_for_event()

    print("\nPart 2 - relapse independent progression\n")
    print("Testing confirmation...")
    test_relapse_independent_confirmation()

    print("Testing baselines...")
    test_relapse_independent_baselines()

    print("Testing minimum increase settings...")
    test_min_increase_settings()

    print("Testing minimal distance...")
    test_relapse_independent_minimal_distance()

    print("Testing first vs. all events...")
    test_relapse_independent_first_vs_all_events()

    print("Testing multiple events re-baselining...")
    test_relapse_independent_multiple_events_rebaselining()

    print("Testing event merging....")
    test_relapse_independent_multiple_events_merging()

    print("\nPart 3 - With relapses\n")
    print("Testing add relapses to dataframe...")
    test_add_relapses_to_follow_up()

    print("Testing post-relapse re-baselining timestamps...")
    test_get_post_relapse_rebaseline_timestamps()

    print("Testing event types...")
    check_event_types()

    print("Testing no improvement w.r.t. PIRA baseline...")
    check_no_improvement_with_respect_to_pira_baseline()

    print("Testing independence of baselines for roving...")
    test_roving_pira_descends_to_general()

    print("Testing RAW/PIRA label assignment for unconfirmed progression...")
    test_raw_pira_unconfirmed()

    print("Testing RAW/PIRA label assignment for confirmed progression...")
    test_raw_pira_confirmed()

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

    print("Testing undefined events annotation modes...")
    test_undefined_event_annotation_options()

    print("Testing event merging with relapses...")
    test_event_merging_with_relapses()

    print("\nAll tests successfully completed.\n")
