"""Functions to annotate study baseline or roving reference to EDSS
follow-up data.

Source publication for roving reference:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6029149/


Roving baseline ambiguities
---------------------------
The original description [1] is vague on the subject. The following
questions are not explicitly answered by the publication:

- Reset baseline only if new value is < study baseline, or if
  new value != study baseline?

- Reset baseline only if a sufficient timedelta has passed since
  a) the last reference assessment or b) the previous assessment?

- Reset baseline only if it is confirmed? Confirm a) if it the new
value is < study baseline or b) always?

- What does confirmation mean for the baseline (especially if the
new baseline is > previous baseline)?

- If a value is repeated, is the first or the last one relevant
as reference when imposing minimal distance conditions?

- If a minimal distance to the reference is required, how are
monotonic decreases before an increase handled?


Our definition of roving baseline
---------------------------------

- Given that the purpose of re-baselining is to increase
  sensitivity, we only re-baseline if a new score is lower 
  than the previous reference.

- We do not impose a minimal distance condition.

- We do not require confirmation for a new roving baseline.


References
----------
[1] Kappos L, Butzkueven H, Wiendl H, Spelman T, Pellegrini F,
Chen Y, Dong Q, Koendgen H, Belachew S, Trojano M; Tysabri® 
Observational Program (TOP) Investigators. “Greater sensitivity 
to multiple sclerosis disability worsening and progression 
events using a roving versus a fixed reference value in a 
prospective cohort study”. In: Multiple Sclerosis Journal
24.7 (2017), pp. 963-973. doi: 10.1177/1352458517709619

"""

import numpy as np
import pandas as pd


def annotate_baseline(
    follow_up_dataframe,
    baseline_type,
    edss_score_column_name,
    time_column_name,
    reference_score_column_name="reference_edss_score",
):
    """Add the relevant reference score and the reference
    timestamp to each row.

    This function returns a copy of the inpt dataframe with
    two additional columns, one that specifies the reference
    EDSS score for each timestep, and one with the timestamp
    of the reference assessment. When using fixed baseline,
    the reference score and the reference timestamp are the
    first score/timestamp in the follow-up.

    The input dataframe is a series of EDSS assessments,
    and must have at least one column for the EDSS score
    and one for the timestamp. The data have to be ordered
    in time, and have only one EDSS score per timestep, i.e.
    time must be strictly monotonically increasing (but does
    not have to start at 0).

    The timestamp has to be provided as int or float, e.g.
    as 'days after baseline' or 'weeks after baseline'. Do
    not provide timestamps as date/datetime.

    Args:
        - follow_up_dataframe: a pandas dataframe
        - baseline_type: 'fixed' or 'roving'; how to define the baseline
        - edss_score_column_name: the name of the column with the scores
        - time_column_name: the name of the column with the timestamps
        - reference_score_column_name: the name of the new column with the
          reference; default is 'reference_edss_score'. The column with the
          reference timestamp will be named reference_score_column_name +
          '_' + time_column name.

    Returns:
        - df: a copy of the input dataframe with the additional columns
          reference_score_column_name and reference_score_column_name +
          '_' + time_column name.

    """
    # Check validity of arguments
    if baseline_type not in ["fixed", "roving"]:
        raise ValueError("Invalid baseline type. Options are 'fixed' or 'roving'.")
    # Check data type of timestamp column
    assert pd.api.types.is_numeric_dtype(
        follow_up_dataframe[time_column_name]
    ), "Timestamps must be numeric, e.g. an integer number of days after baseline."
    # Check if input data are well ordered and timestamps are unambiguous
    assert (
        follow_up_dataframe[time_column_name].is_monotonic_increasing
        and follow_up_dataframe[time_column_name].is_unique
    ), "Input data are not well ordered or contain ambiguous timestamps."

    # Prepare df to return in the end
    annotated_df = follow_up_dataframe.copy()
    annotated_df[reference_score_column_name] = np.NaN

    baseline_score = follow_up_dataframe.iloc[0][edss_score_column_name]
    baseline_timestamp = follow_up_dataframe.iloc[0][time_column_name]

    # If we use fixed baseline, we're done:
    if baseline_type == "fixed":
        indices = follow_up_dataframe.iloc[1:].index.values
        annotated_df.loc[indices, reference_score_column_name] = baseline_score
        annotated_df.loc[
            indices, reference_score_column_name + "_" + time_column_name
        ] = baseline_timestamp

    # For roving baseline, we have to loop over the follow-up.
    # NOTE: without keeping track of the reference timestamp, we could
    # just take the minimum EDSS of all previous values as our reference...
    # We don't check the first (study baseline anyway) and last (irrelevant) row.
    # NOTE: iterrows returns row labels, not positions.
    if baseline_type == "roving":
        current_roving_reference = baseline_score
        current_roving_reference_timestamp = baseline_timestamp
        for i, row in follow_up_dataframe.iloc[:-1].iterrows():
            current_edss = row[edss_score_column_name]
            current_timestamp = row[time_column_name]

            # Check if the new score qualifies
            if current_edss < current_roving_reference:
                current_roving_reference = current_edss
                current_roving_reference_timestamp = current_timestamp

            # Write results
            annotated_df.at[
                i + 1, reference_score_column_name
            ] = current_roving_reference
            annotated_df.at[
                i + 1, reference_score_column_name + "_" + time_column_name
            ] = current_roving_reference_timestamp

    return annotated_df


if __name__ == "__main__":
    # Usage example
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

    annotated_example_follow_up_df = annotate_baseline(
        follow_up_dataframe=example_follow_up_df,
        baseline_type="roving",
        edss_score_column_name="edss_score",
        time_column_name="days_after_baseline",
        reference_score_column_name="reference_edss_score",
    )

    print(annotated_example_follow_up_df)
