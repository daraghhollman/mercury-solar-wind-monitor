import datetime as dt
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import polars as pl
import requests
from sunpy.time import TimeRange

from mvswm.data import (
    get_helios_data,
    get_messenger_data,
    get_parker_data,
    get_solar_orbiter_data,
)

CACHE_DIR = Path(".cache")

# Loaders to fetch data from the entire mission at once
MAG_LOADERS: Dict[str, Callable] = {
    "MESSENGER": partial(
        get_messenger_data,
        # time_range=TimeRange("2011-03-23", "2015-04-30"),
        time_range=TimeRange("2011-03-23", "2011-04-30"),
        product="MAG",
    ),
    "Solar Orbiter": partial(
        get_solar_orbiter_data,
        time_range=TimeRange("2020-02-11", "2026-01-01"),
        product="mag-rtn-normal-1-minute",
        quality_limit=2,
    ),
    "Parker Solar Probe": partial(
        get_parker_data,
        time_range=TimeRange("2018-08-13", "2025-11-01"),
        product="psp-fld-l2-mag-rtn-1min",
    ),
    "Helios 1": partial(
        get_helios_data,
        time_range=TimeRange("1974-12-11", "1985-09-05"),
        spacecraft=1,
    ),
    "Helios 2": partial(
        get_helios_data,
        time_range=TimeRange("1976-01-16", "1980-03-09"),
        spacecraft=2,
    ),
}


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
