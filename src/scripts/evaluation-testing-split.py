"""
Splits MESSENGER cruise data into two training datasets:
    - Evalutation, to iterrate on when constructing the model
    - Testing, to evaluate a final performace metric on unseen data
"""

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from sunpy.time import TimeRange

from mvswm.data import Spacecraft
from mvswm.utils.colours import BLUE, RED

MESSENGER_ORBIT_START = dt.datetime(2011, 3, 21)
MISSION_START_BUFFER = dt.timedelta(days=30)

SAVE_DIR = Path(__file__).parents[2] / "data/"


def main() -> None:
    data = Spacecraft("MESSENGER").data

    # Skip early cruise phase to increase coverage.
    data = data.filter(pl.col("UTC") > dt.datetime(2008, 1, 1))

    # Clean data by removing flybys of Earth, Venus, and Mercury
    data = remove_flybys(data)

    # Clean data by removing unphysical instrument spikes in MAG data.
    data = remove_mag_spikes(data)

    # NOTE: Do we need to do this?
    # Clean data by removing anything outside Mercury heliocentric distances
    # data = data.filter(pl.col("Radius [au]") < 0.47)

    _set_names = ["Evaluation", "Testing", "Application"]

    # Remove orbit phase from all data
    data_to_split = data.filter(
        pl.col("UTC") <= MESSENGER_ORBIT_START - MISSION_START_BUFFER
    )

    evaluation_data, testing_data = split_sequentially(
        data_to_split, chunk_length=dt.timedelta(days=88), ratio=(2, 1)
    )

    # Insert nulls where gaps exist
    evaluation_data = make_gaps_null(evaluation_data)
    testing_data = make_gaps_null(testing_data)

    # A figure to view the data within the chunks
    fig, ax = plt.subplots()

    ax.plot(evaluation_data["UTC"], evaluation_data["Radius [au]"], color=RED)
    ax.plot(testing_data["UTC"], testing_data["Radius [au]"], color=BLUE)

    plt.show()

    # Ensure savedir exists
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Save datasets
    print(f"Writing to {SAVE_DIR}")
    evaluation_data.write_csv(SAVE_DIR / "evaluation-data.csv")
    testing_data.write_csv(SAVE_DIR / "testing-data.csv")


def split_sequentially(
    data_to_split: pl.DataFrame,
    chunk_length: dt.timedelta,
    ratio: tuple[int, int] = (1, 1),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splitting evaluation and testing by separating chunks of fixed length,
    in a repeating pattern determined by `ratio`.

    `ratio` is (evaluation_chunks, testing_chunks). For example, (3, 1) gives
    a repeating pattern of three evaluation chunks followed by one testing
    chunk, i.e. a 75/25 split.
    """

    # Check inputs are correct
    evalulation_weight, testing_weight = ratio
    if evalulation_weight < 1 or testing_weight < 1:
        raise ValueError(f"Both parts of ratio must be positive integers, got {ratio}")

    cycle_length = evalulation_weight + testing_weight

    # Start from the begining, and separate out chunks.
    start_time = data_to_split["UTC"][0]

    chunk_index = (
        pl.col("UTC") - start_time
    ).dt.total_seconds() // chunk_length.total_seconds()

    # Position of each chunk within the repeating cycle; the first
    # `eval_chunks` positions go to evaluation, the remainder to testing.
    chunked_mask = (chunk_index % cycle_length) < evalulation_weight

    evaluation_data = data_to_split.filter(chunked_mask)
    testing_data = data_to_split.filter(~chunked_mask)

    return evaluation_data, testing_data


def remove_flybys(data: pl.DataFrame) -> pl.DataFrame:

    _data_cadence = data["UTC"][1] - data["UTC"][0]

    # Clean data by removing flybys of Earth, Mercury, and Venus
    flybys: list[TimeRange] = [
        TimeRange("2005-08-01", "2005-08-03"),  # Earth
        TimeRange("2006-10-23", "2006-10-25"),  # Venus 1
        TimeRange("2007-06-04", "2007-06-06"),  # Venus 2
        TimeRange("2008-01-13", "2008-01-15"),  # Mercury 1
        TimeRange("2008-10-05", "2008-10-07"),  # Mercury 2
        TimeRange("2009-09-28", "2009-09-30"),  # Mercury 3
    ]
    data = data.filter(
        ~pl.any_horizontal(
            pl.col("UTC").is_between(flyby.start.to_datetime(), flyby.end.to_datetime())
            for flyby in flybys
        )
    )

    return data


def remove_mag_spikes(
    data: pl.DataFrame,
    threshold: float = 10_000,
    buffer: int = 1,  # rows
    col: str = "|B| [nT]",
) -> pl.DataFrame:
    """Drop rows where `col` exceeds `threshold`, plus `buffer` rows either side."""
    is_spike = (pl.col(col) > threshold).fill_null(False)

    # A row is bad if any spike lies within +/- buffer rows (centered rolling max)
    bad = (
        is_spike.cast(pl.Int8)
        .rolling_max(window_size=2 * buffer + 1, center=True, min_samples=1)
        .cast(pl.Boolean)
    )
    return data.filter(~bad)


def make_gaps_null(
    df: pl.DataFrame,
    time_col: str = "UTC",
    max_gap: dt.timedelta = dt.timedelta(hours=1),
) -> pl.DataFrame:
    """Insert a null row inside any gap larger than max_gap."""
    df = df.sort(time_col)
    gap_rows = df.filter(
        (pl.col(time_col).shift(-1) - pl.col(time_col)) > max_gap
    ).select((pl.col(time_col) + max_gap / 2).cast(df.schema[time_col]))
    return pl.concat([df, gap_rows], how="diagonal").sort(time_col)


if __name__ == "__main__":
    main()
