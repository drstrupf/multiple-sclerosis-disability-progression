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


def test_is_large_enough_increase_or_decrease():
    # Part 1 - Accrual annotation mode
    # Test 1 - minimum required increase + 0.5 irrespective of reference
    test_cases_1 = [
        # Increase
        {"current": 0, "reference": 0, "target": [False, False]},
        {"current": 0.5, "reference": 0, "target": [True, False]},
        {"current": 1.0, "reference": 0, "target": [True, False]},
        {"current": 1.5, "reference": 0, "target": [True, False]},
        {"current": 2.0, "reference": 2.0, "target": [False, False]},
        {"current": 2.5, "reference": 2.0, "target": [True, False]},
        {"current": 3.0, "reference": 2.0, "target": [True, False]},
        # Decrease, should yield False irrespective of delta
        {"current": 5.5, "reference": 6.0, "target": [False, False]},
        {"current": 5.0, "reference": 6.0, "target": [False, False]},
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
        err_msg="Accrual mode: minimal increase + 0.5 irrespective of baseline failed!",
    )
    # Test 2 - minimum required increase + 1.0 irrespective of reference
    test_cases_2 = [
        {"current": 0, "reference": 0, "target": [False, False]},
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [True, False]},
        {"current": 1.5, "reference": 0, "target": [True, False]},
        {"current": 2.0, "reference": 2.0, "target": [False, False]},
        {"current": 2.5, "reference": 2.0, "target": [False, False]},
        {"current": 3.0, "reference": 2.0, "target": [True, False]},
        {"current": 10.0, "reference": 9.5, "target": [False, False]},
        # Decrease, should yield False irrespective of delta
        {"current": 5.5, "reference": 6.0, "target": [False, False]},
        {"current": 5.0, "reference": 6.0, "target": [False, False]},
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
        err_msg="Accrual mode: minimal increase + 1.0 irrespective of baseline failed!",
    )
    # Test 3 - minimum required increase from 0 + 1.5
    test_cases_3 = [
        {"current": 0, "reference": 0, "target": [False, False]},
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [False, False]},
        {"current": 1.5, "reference": 0, "target": [True, False]},
        {"current": 2.0, "reference": 2.0, "target": [False, False]},
        {"current": 2.5, "reference": 2.0, "target": [True, False]},
        {"current": 3.0, "reference": 2.0, "target": [True, False]},
        # Decrease, should yield False irrespective of delta
        {"current": 0.0, "reference": 1.5, "target": [False, False]},
        {"current": 0.0, "reference": 2.0, "target": [False, False]},
        {"current": 5.5, "reference": 6.0, "target": [False, False]},
        {"current": 5.0, "reference": 6.0, "target": [False, False]},
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
        err_msg="Accrual mode: minimal increase + 1.5 from 0 failed!",
    )
    # Test 4 - standard case
    test_cases_4 = [
        {"current": 0, "reference": 0, "target": [False, False]},
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
        # Decrease, should yield False irrespective of delta
        {"current": 0.0, "reference": 1.5, "target": [False, False]},
        {"current": 0.0, "reference": 2.0, "target": [False, False]},
        {"current": 5.5, "reference": 6.0, "target": [False, False]},
        {"current": 5.0, "reference": 6.0, "target": [False, False]},
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
        err_msg="Accrual mode: standard minimal increase settings failed!",
    )

    # Part 2 - experimental-inverted annotation mode
    # Test 1 - minimum required decrease + 0.5 irrespective of reference
    test_cases_1 = [
        # Increase, False irrespective of delta
        {"current": 0, "reference": 0, "target": [False, False]},
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [False, False]},
        # Decrease
        {"current": 0, "reference": 0.5, "target": [False, True]},
        {"current": 0, "reference": 1.0, "target": [False, True]},
        {"current": 0, "reference": 1.5, "target": [False, True]},
        {"current": 9.5, "reference": 10.0, "target": [False, True]},
        {"current": 5.5, "reference": 6.0, "target": [False, True]},
        {"current": 5.0, "reference": 6.0, "target": [False, True]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    annotation_mode="experimental-inverted",
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
        err_msg="Inverted mode: minimal decrease + 0.5 irrespective of baseline failed!",
    )
    # Test 2 - minimum required decrease + 1.0 irrespective of reference
    test_cases_2 = [
        # Increase, False irrespective of delta
        {"current": 0, "reference": 0, "target": [False, False]},
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [False, False]},
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
                    annotation_mode="experimental-inverted",
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
        err_msg="Inverted mode: minimal decrease + 1.0 irrespective of baseline failed!",
    )
    # Test 3 - minimum required increase from 0 + 1.5
    test_cases_3 = [
        # Increase, False irrespective of delta
        {"current": 0, "reference": 0, "target": [False, False]},
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [False, False]},
        # Decrease
        {"current": 0, "reference": 0.5, "target": [False, False]},
        {"current": 0, "reference": 1.0, "target": [False, False]},
        {"current": 0, "reference": 1.5, "target": [False, True]},
        {"current": 9.5, "reference": 10.0, "target": [False, True]},
        {"current": 5.5, "reference": 6.0, "target": [False, True]},
        {"current": 5.0, "reference": 6.0, "target": [False, True]},
        {"current": 5.0, "reference": 5.5, "target": [False, False]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    annotation_mode="experimental-inverted",
                    opt_max_score_that_requires_plus_1=5,
                    opt_larger_increment_from_0=True,
                )._is_large_enough_increase_or_decrease(
                    current_edss=test_case["current"],
                    reference_edss=test_case["reference"],
                )
                for test_case in test_cases_3
            ]
        ),
        np.array([test_case["target"] for test_case in test_cases_3]),
        err_msg="Inverted mode: minimal decrease + 1.5 from 0 failed!",
    )
    # Test 4 - standard case
    test_cases_4 = [
        # Increase, False irrespective of delta
        {"current": 0, "reference": 0, "target": [False, False]},
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [False, False]},
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
                edssannotation.EDSSAnnotation(
                    annotation_mode="experimental-inverted",
                )._is_large_enough_increase_or_decrease(
                    current_edss=test_case["current"],
                    reference_edss=test_case["reference"],
                )
                for test_case in test_cases_4
            ]
        ),
        np.array([test_case["target"] for test_case in test_cases_4]),
        err_msg="Inverted mode: standard minimal decrease settings failed!",
    )

    # Part 3 - experimental-symmetric annotation mode
    # Test 1 - minimum required decrease + 0.5 irrespective of reference
    test_cases_1 = [
        # Increase
        {"current": 0, "reference": 0, "target": [False, False]},
        {"current": 0.5, "reference": 0, "target": [True, False]},
        {"current": 1.0, "reference": 0, "target": [True, False]},
        {"current": 1.5, "reference": 0, "target": [True, False]},
        # Decrease
        {"current": 0, "reference": 0.5, "target": [False, True]},
        {"current": 0, "reference": 1.0, "target": [False, True]},
        {"current": 0, "reference": 1.5, "target": [False, True]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    annotation_mode="experimental-symmetric",
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
        err_msg="Symmetric mode: minimal delta +- 0.5 irrespective of baseline failed!",
    )
    # Test 2 - minimum required decrease + 1.0 irrespective of reference
    test_cases_2 = [
        # Increase
        {"current": 0, "reference": 0, "target": [False, False]},
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [True, False]},
        {"current": 1.5, "reference": 0, "target": [True, False]},
        # Decrease
        {"current": 0, "reference": 0.5, "target": [False, False]},
        {"current": 0, "reference": 1.0, "target": [False, True]},
        {"current": 0, "reference": 1.5, "target": [False, True]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    annotation_mode="experimental-symmetric",
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
        err_msg="Symmetric mode: minimal delta +- 1.0 irrespective of baseline failed!",
    )
    # Test 3 - minimum required increase from 0 + 1.5
    test_cases_3 = [
        # Increase
        {"current": 0, "reference": 0, "target": [False, False]},
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [False, False]},
        {"current": 1.5, "reference": 0, "target": [True, False]},
        # Decrease
        {"current": 0, "reference": 0.5, "target": [False, False]},
        {"current": 0, "reference": 1.0, "target": [False, False]},
        {"current": 0, "reference": 1.5, "target": [False, True]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    annotation_mode="experimental-symmetric",
                    opt_max_score_that_requires_plus_1=5,
                    opt_larger_increment_from_0=True,
                )._is_large_enough_increase_or_decrease(
                    current_edss=test_case["current"],
                    reference_edss=test_case["reference"],
                )
                for test_case in test_cases_3
            ]
        ),
        np.array([test_case["target"] for test_case in test_cases_3]),
        err_msg="Symmetric mode: minimal increase + 1.5 from 0 failed!",
    )
    # Test 4 - standard case
    test_cases_4 = [
        # Increase
        {"current": 0, "reference": 0, "target": [False, False]},
        {"current": 0.5, "reference": 0, "target": [False, False]},
        {"current": 1.0, "reference": 0, "target": [False, False]},
        {"current": 1.5, "reference": 0, "target": [True, False]},
        {"current": 5.5, "reference": 5.0, "target": [False, False]},
        {"current": 6.0, "reference": 5.0, "target": [True, False]},
        {"current": 6.0, "reference": 5.5, "target": [True, False]},
        {"current": 6.5, "reference": 5.5, "target": [True, False]},
        # Decrease
        {"current": 0, "reference": 0.5, "target": [False, False]},
        {"current": 0, "reference": 1.0, "target": [False, False]},
        {"current": 0, "reference": 1.5, "target": [False, True]},
        {"current": 5.5, "reference": 6.0, "target": [False, True]},
        {"current": 5.0, "reference": 6.0, "target": [False, True]},
        {"current": 5.0, "reference": 5.5, "target": [False, False]},
        {"current": 4.5, "reference": 5.5, "target": [False, True]},
        {"current": 4.5, "reference": 5.0, "target": [False, False]},
    ]
    np.testing.assert_array_equal(
        np.array(
            [
                edssannotation.EDSSAnnotation(
                    annotation_mode="experimental-symmetric",
                )._is_large_enough_increase_or_decrease(
                    current_edss=test_case["current"],
                    reference_edss=test_case["reference"],
                )
                for test_case in test_cases_4
            ]
        ),
        np.array([test_case["target"] for test_case in test_cases_4]),
        err_msg="Symmetric mode: standard minimal increase settings failed!",
    )


if __name__ == "__main__":
    print("\nPart 1 - building blocks\n")
    print("Testing 'is_above_progress_threshold'...")
    test_is_large_enough_increase_or_decrease()

    print("\nAll tests successfully completed.\n")
