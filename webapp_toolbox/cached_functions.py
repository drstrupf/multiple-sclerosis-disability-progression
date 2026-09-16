"""Functions to process uploaded files."""

import pandas as pd

from tools import preprocessing


def load_excel_table(uploaded_file):
    return uploaded_file


def preprocess_edss_follow_up(uploaded_follow_up_file):
    raw_follow_ups = pd.read_excel(uploaded_follow_up_file)
    raw_follow_ups["surrogate_id"] = 0
    processed_follow_ups = preprocessing.prepare_follow_ups(
        follow_ups_dataframe=raw_follow_ups,
        max_days_between_timestamps=1_000_000,
        min_days_overall=0,
        min_n_timestamps=0,
        id_column_name="surrogate_id",
        date_column_name="edss_date",
        days_after_baseline_column_name="days_after_baseline",
    )
    return processed_follow_ups


def preprocess_sync_relapse_timestamps(uploaded_relapses_file, processed_follow_ups):
    raw_relapses = pd.read_excel(uploaded_relapses_file)
    raw_relapses["surrogate_id"] = 0
    processed_relapses = preprocessing.sync_relapse_data_to_follow_ups(
        preprocessed_follow_ups_dataframe=processed_follow_ups.rename(
            columns={"edss_date": "date"}
        ),
        relapses_dataframe=raw_relapses.rename(columns={"relapse_date": "date"}),
        id_column_name="surrogate_id",
        date_column_name="date",
        days_after_baseline_column_name="days_after_baseline",
        block_baseline_flag_column_name="is_block_baseline",
    )
    return processed_relapses[
        ["relapse_date", "surrogate_id", "block_id", "days_after_baseline"]
    ]


if __name__ == "__main__":
    pass
