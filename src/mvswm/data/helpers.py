import datetime as dt
from pathlib import Path

import polars as pl
import requests
from sunpy.time import TimeRange

from mvswm.data.spacecraft import Spacecraft

CACHE_DIR = Path(".cache")


def load_crossings():

    # Download the Hollman et al. (2026) crossing list
    url = "https://zenodo.org/records/17814795/files/hollman_2025_crossing_list.csv?download=1"
    crossing_list_path = CACHE_DIR / "hollman_2026_crossing_list.csv"

    # If the file doesn't exist, download it
    if not crossing_list_path.exists():
        response = requests.get(url)
        with open(crossing_list_path, "wb") as file:
            file.write(response.content)

    return pl.read_csv(crossing_list_path, try_parse_dates=True).rename({"Time": "UTC"})


def filter_messenger_mag(
    data: pl.DataFrame, buffer: dt.timedelta = dt.timedelta(0)
) -> pl.DataFrame:

    messenger_crossings = load_crossings()

    solar_wind_intervals = (
        messenger_crossings.filter(pl.col("Label") == "BS_OUT")
        .rename({"UTC": "Start Time"})
        .join_asof(
            messenger_crossings.filter(pl.col("Label") == "BS_IN").rename(
                {"UTC": "End Time"}
            ),
            left_on="Start Time",
            right_on="End Time",
            strategy="forward",  # BS_IN after each BS_OUT
        )
        .select("Start Time", "End Time")
        .drop_nulls()
        # Add buffer
        .with_columns(
            [
                (pl.col("Start Time") + buffer).alias("Start Time"),
                (pl.col("End Time") - buffer).alias("End Time"),
            ]
        )
        # drop intervals that the buffer has consumed entirely
        .filter(pl.col("Start Time") < pl.col("End Time"))
    )

    filtered_data = (
        data.with_columns(pl.col("UTC").dt.cast_time_unit("us"))
        .sort("UTC")
        .join_asof(
            solar_wind_intervals.sort("Start Time"),
            left_on="UTC",
            right_on="Start Time",
            strategy="backward",  # find the most recent BS_OUT before each row
        )
        .with_columns(
            pl.when(pl.col("UTC") < pl.col("End Time"))
            .then(pl.col(col))
            .otherwise(None)
            .alias(col)
            for col in data.columns
            if col != "UTC"
        )
        .select(data.columns)
    )

    return filtered_data


def get_messenger_solar_wind_data(
    time_range: TimeRange, bow_shock_buffer: dt.timedelta
) -> pl.DataFrame:
    """
    Loads MESSENGER MAG data and filters to only times in the solar wind based
    on bow shock crossings by Hollman et al. (2026). These crossings are
    buffered to ensure the data is solar wind.

    Params
    ------
    time_range: TimeRange
        The range of data to load

    bow_shock_buffer: datetime.timedelta
        How much time to buffer all bow shock crossings by

    Returns
    -------
    data: polars.DataFrame
    """

    messenger = Spacecraft("MESSENGER")
    data = messenger.get_data_in_range(time_range)

    # Remove time within bow shock. This is buffered
    data = filter_messenger_mag(data, buffer=bow_shock_buffer)

    return data
