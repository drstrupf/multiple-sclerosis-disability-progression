"""This module is a collection of functions for input data preprocessing.

Note that data preprocessing (from datetime to integer unit after baseline)
is not part of the EDSS disability accrual annotation algorithm, since it
can be tricky if source data are of low quality (inconsistent datetime formats,
duplicate entries, not ordered properly...). The functions in this module
are meant as an example of how to do the preprocessing, and they are built to
just throw errors if the input data do not satisfy some minimal formatting
requirements. They do not try to fix anything, they don't even re-order data...

In addition, there might be more elegant ways to do this :)

"""

import numpy as np
import pandas as pd


def check_input_date_data(input_dataframe, date_column_name, verbose=False):
    """Check input data format.

    This function raises VauleErrors if the timestamps provided contain
    a time part, are not unique, or are not ordered (ascending).

    Args:
        - input_dataframe: a pandas dataframe with at least a date column
        - date_column_name: the name of the date column
        - verbose: display additional messages

    Returns: None

    """
    # Part 1 - make sure the dates do not contain a time part.
    if verbose:
        print("Check date format.")
    if (
        pd.to_datetime(input_dataframe[date_column_name]).dt.floor("d")
        == pd.to_datetime(input_dataframe[date_column_name])
    ).all():
        if verbose:
            print("Date only, proceed.")
        pass
    else:
        if verbose:
            print("Date contains time part.")
        raise ValueError("Date information contains time part.")
    # Part 2 - check whether timestamps are strictly monotonically increasing.
    if verbose:
        print("\nCheck order and uniqueness of timestamps.")
    if (
        input_dataframe[date_column_name].is_monotonic_increasing
        and input_dataframe[date_column_name].is_unique
    ):
        if verbose:
            print("Unique and well ordered.")
        pass
    elif input_dataframe[date_column_name].is_monotonic_increasing and (
        not input_dataframe[date_column_name].is_unique
    ):
        if verbose:
            print("Well ordered, but not unique.")
        raise ValueError("Input data contain duplicate timestamps.")
    elif not (input_dataframe[date_column_name].is_monotonic_increasing) and (
        input_dataframe[date_column_name].is_unique
    ):
        if verbose:
            print("Unique, but not well ordered.")
        raise ValueError("Input data are not well ordered.")
    else:
        if verbose:
            print("Not unique and a mess.")
        raise ValueError(
            "Input data contain duplicate timestamps and are not well ordered."
        )


def find_connected_blocks(
    input_dataframe,
    max_days_between_timestamps,
    min_days_overall,
    min_n_timestamps,
    date_column_name,
    block_id_start=0,
    block_id_column_name="block_id",
    block_baseline_flag_column_name="is_block_baseline",
    days_after_baseline_column_name="days_after_baseline",
    tests_verbose=False,
):
    """Assign unique integer IDs to connected follow-up blocks.

    Dates must not contain time information, and must be ordered and unique.
    If these requirements are not met, the function will raise a ValueError,
    and will NOT (try to) fix anything. Data quality issues need to be solved
    by a subject matter expert.

    Args:
        - input_dataframe: a Pandas dataframe with time series data, ordered by time
        - max_days_between_timestamps: maximum number of days between two assessments
        - min_days_overall: the minimal follow-up period
        - min_n_timestamps: the minimal number of assessments
        - date_column_name: column where the assessment date is specified
        - block_id_start: block enumeration starts at this number
        - block_id_column_name: name of the column that holds the block ID
        - block_baseline_flag_column_name: name of the column that holds the 'is baseline' flag
        - days_after_baseline_column_name: column where days after block baseline will be written
        - tests_verbose: show additional messages from quality control

    Returns:
        - df: the original dataframe, with columns for the block IDs, the block baseline flags,
              and numeric timestamps (float)

    """
    # Check input data - raises an error if not ok.
    check_input_date_data(
        input_dataframe=input_dataframe,
        date_column_name=date_column_name,
        verbose=tests_verbose,
    )
    # Create a working copy
    return_df = input_dataframe.copy()
    # Check if assessment is close enough to previous
    return_df["previous"] = return_df[date_column_name].shift(1)
    return_df["previous_in_time"] = return_df.apply(
        lambda row: (
            True
            if row[date_column_name] - row["previous"]
            <= pd.Timedelta(max_days_between_timestamps, unit="days")
            else False
        ),
        axis=1,
    )
    # Check if assessment is close enough to next
    return_df["next"] = return_df[date_column_name].shift(-1)
    return_df["next_in_time"] = return_df.apply(
        lambda row: (
            True
            if row["next"] - row[date_column_name]
            <= pd.Timedelta(max_days_between_timestamps, unit="days")
            else False
        ),
        axis=1,
    )
    # Flag potential block starts
    return_df["is_potential_block_start"] = return_df.apply(
        lambda row: (
            True if (not row["previous_in_time"]) and (row["next_in_time"]) else False
        ),
        axis=1,
    )
    # Get the indices of the blocks
    split_ids = [
        idx for idx, row in return_df.iterrows() if row["is_potential_block_start"]
    ]
    candidate_block_coordinates = [
        (i, j)
        for i, j in zip(
            split_ids, [split_id - 1 for split_id in split_ids[1:]] + [None]
        )
    ]
    # Slice out the blocks
    candidate_blocks = [
        return_df.loc[pair[0] : pair[1]] for pair in candidate_block_coordinates
    ]
    # A block could contain values at the end that are too far
    # away from the previous assessment; remove them.
    candidate_blocks_clean = [
        block[(block["previous_in_time"]) | (block["is_potential_block_start"])].copy()
        for block in candidate_blocks
    ]

    # Check if block covers minimal required follow-up period.
    # Define a helper function
    def _coverage_conditions_satisfied(block_df):
        block_start_date = block_df.iloc[0][date_column_name]
        block_end_date = block_df.iloc[-1][date_column_name]
        if (
            block_end_date - block_start_date
            >= pd.Timedelta(min_days_overall, unit="days")
        ) and len(block_df) >= min_n_timestamps:
            return True
        else:
            return False

    # Check each block
    block_flags = [
        _coverage_conditions_satisfied(
            block_df=block,
        )
        for block in candidate_blocks_clean
    ]
    # Get a list of all valid blocks
    valid_blocks = [
        candidate_blocks_clean[i] for i, flag in enumerate(block_flags) if flag
    ]
    # Assign unique block IDs and flag block baseline
    return_df[block_id_column_name] = np.nan
    return_df[block_baseline_flag_column_name] = False
    return_df[days_after_baseline_column_name] = np.nan
    for i, block in enumerate(valid_blocks):
        first = min([j for j, _ in block.iterrows()])
        for j, block_row in block.iterrows():
            if j == first:
                return_df.at[j, block_baseline_flag_column_name] = True
                return_df.at[j, days_after_baseline_column_name] = 0
                block_start_date = block_row[date_column_name]
            return_df.at[j, block_id_column_name] = block_id_start + i
            return_df.at[j, days_after_baseline_column_name] = (
                block_row[date_column_name] - block_start_date
            ).days
    # Drop helper columns before returning
    return return_df.drop(
        columns=[
            "previous",
            "previous_in_time",
            "next",
            "next_in_time",
            "is_potential_block_start",
        ]
    )


def prepare_follow_ups(
    follow_ups_dataframe,
    max_days_between_timestamps,
    min_days_overall,
    min_n_timestamps,
    id_column_name,
    date_column_name,
    days_after_baseline_column_name="days_after_baseline",
    block_id_column_name="block_id",
    block_baseline_flag_column_name="is_block_baseline",
    tests_verbose=False,
):
    """TBD"""
    follow_up_ids = follow_ups_dataframe[id_column_name].drop_duplicates()
    # Set up a list where we collect the blocks, and an overall ID
    follow_ups_with_blocks = []
    block_id_overall = 0
    # Find the blocks for each follow-up
    for follow_up_id in follow_up_ids:
        follow_up_with_blocks = find_connected_blocks(
            input_dataframe=follow_ups_dataframe[
                follow_ups_dataframe[id_column_name] == follow_up_id
            ],
            max_days_between_timestamps=max_days_between_timestamps,
            min_days_overall=min_days_overall,
            min_n_timestamps=min_n_timestamps,
            block_id_start=block_id_overall,
            date_column_name=date_column_name,
            days_after_baseline_column_name=days_after_baseline_column_name,
            block_id_column_name=block_id_column_name,
            block_baseline_flag_column_name=block_baseline_flag_column_name,
            tests_verbose=tests_verbose,
        )
        max_block_id = follow_up_with_blocks[block_id_column_name].max()
        if max_block_id >= 0:
            block_id_overall = max_block_id + 1
        follow_ups_with_blocks = follow_ups_with_blocks + [follow_up_with_blocks]
    # Concatenate and drop assessments that are not part of a connected follow-up
    follow_up_with_blocks_df = (
        pd.concat(follow_ups_with_blocks)
        .dropna(subset=[block_id_column_name])
        .reset_index(drop=True)
    )
    # Format timestamps and IDs to integer
    follow_up_with_blocks_df[days_after_baseline_column_name] = (
        follow_up_with_blocks_df[days_after_baseline_column_name].astype(int)
    )
    follow_up_with_blocks_df[block_id_column_name] = follow_up_with_blocks_df[
        block_id_column_name
    ].astype(int)
    return follow_up_with_blocks_df


def sync_relapse_data_to_follow_ups(
    preprocessed_follow_ups_dataframe,
    relapses_dataframe,
    id_column_name,
    date_column_name,
    days_after_baseline_column_name="days_after_baseline",
    block_baseline_flag_column_name="is_block_baseline",
    tests_verbose=False,
):
    """TBD"""
    # Check input data - raises an error if not ok.
    for follow_up_id in relapses_dataframe[id_column_name].drop_duplicates():
        check_input_date_data(
            input_dataframe=relapses_dataframe[
                relapses_dataframe[id_column_name] == follow_up_id
            ],
            date_column_name=date_column_name,
            verbose=tests_verbose,
        )
    # Get all block baseline dates - this will never be empty,
    # since we use this on preprocessed follow-ups.
    block_baselines = preprocessed_follow_ups_dataframe[
        preprocessed_follow_ups_dataframe[block_baseline_flag_column_name]
    ]
    # Add the relapses to the baselines (one baseline, zero to many
    # relapses). Note that we use the default inner join and do not
    # specify 'on'. If 'on' is None and not merging on indexes then
    # this defaults to the intersection of the columns in both DataFrames.
    # By renaming the date and days after baseline columns, we make
    # sure that they are not used as merge keys. If a follow-up ID is
    # provided, it is used as a merge key.
    synced_relapses = pd.merge(
        left=block_baselines.drop(columns=[days_after_baseline_column_name]),
        right=relapses_dataframe.rename(
            columns={date_column_name: "relapse_" + date_column_name}
        ),
    )
    # Compute days after block baseline for each relapse
    synced_relapses[days_after_baseline_column_name] = (
        synced_relapses["relapse_" + date_column_name]
        - synced_relapses[date_column_name]
    ).dt.days
    # Clean up
    synced_relapses = synced_relapses.rename(
        columns={date_column_name: "block_baseline_" + date_column_name}
    ).drop(columns=[block_baseline_flag_column_name])
    return synced_relapses


if __name__ == "__main__":
    pass
