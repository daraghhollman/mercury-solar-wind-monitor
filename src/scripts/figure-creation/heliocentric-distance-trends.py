"""
Scatter B-field for evaluation, validation, and application data sets as a
function of heliocentric distance.
"""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LogNorm, SymLogNorm
from sunpy.time import TimeRange

from mvswm.data import Spacecraft, filter_messenger_mag
from mvswm.utils.colours import BLACK, BLUE, RED

# TODO: needs refining
MESSENGER_ORBIT_START = dt.datetime(2011, 3, 21)
MISSION_START_BUFFER = dt.timedelta(days=30)

X_EDGES = np.linspace(0.29, 0.48, 100)
Y_EDGES = np.linspace(0, 80, 80)
BINS = [X_EDGES, Y_EDGES]


def main() -> None:

    data = Spacecraft("MESSENGER").data

    # Skip early cruise phase to increase coverage.
    data = data.filter(pl.col("UTC") > dt.datetime(2008, 8, 1))

    # Clean data by removing flybys of Earth, Venus, and Mercury
    # TODO: No flybys are removed with this new early cruise phase skip. Check
    # if they all happened before august 2008
    data = remove_flybys(data)

    # Clean data by removing anything outside Mercury heliocentric distances
    data = data.filter(pl.col("Radius [au]") < 0.47)

    # Need to set up a training and testing split based on dates
    #
    # 'Evaluation' will mean that these data are fit to while determning the
    # best architecture and parameters of model.
    #
    # 'Testing' will mean that these data will be fit to after the evaluation
    # to determine the performance of the technique.
    #
    # 'Application' will mean that these data will be used to finally apply the
    # model - but no artifical gaps / performance evaluation will take place.
    #
    # We need to be careful in splitting up these data as to not introduce
    # trends. i.e. we need an even spread in solar cycle phase, heliocentric
    # distance, etc.
    #
    # Additionally, we also want to introduce exclusion criteria - i.e. if our
    # heliocentric distance is outside of Mercury's range.

    set_names = ["Evaluation", "Testing", "Application"]

    data_to_split = data.filter(pl.col("UTC") <= MESSENGER_ORBIT_START - MISSION_START_BUFFER)
    application_data = data.filter(pl.col("UTC") > MESSENGER_ORBIT_START)

    # Clean data by removing MESSENGER magnetosphere times
    application_data = remove_non_solar_wind(
        application_data, buffer=dt.timedelta(minutes=10)
    )

    # Split description
    # split_date = dt.datetime(2010, 1, 1)
    # evaluation_data, testing_data = split_trivially(data_to_split, split_date)
    # split_description = f"Splitting Evaluation and Testing on date: {split_date}"

    chunk_length = dt.timedelta(days=27)
    evaluation_data, testing_data = split_sequentially(
        data_to_split, chunk_length=chunk_length
    )
    split_description = (
        f"Splitting Evaluation and Testing sequentially in chunks of {chunk_length}"
    )
    # stats = chunk_length_stats(data_to_split, chunk_length)
    # print(stats)

    # Plotting
    # Two columns of axes, one for the data, another to compare with the
    # average of all 3 rows. Each row is a different dataset
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)

    datasets = [evaluation_data, testing_data, application_data]
    histograms = [
        # Density = True to normalise by area and allow comparison of datasets with different sizes.
        np.histogram2d(d["Radius [au]"], d["|B| [nT]"], bins=BINS, density=True)[0]
        for d in datasets
    ]

    data_norm = LogNorm(vmin=0.01, vmax=np.max(np.array(histograms)))

    # Data axes
    for i, hist in enumerate(histograms):

        ax = axes[i, 0]

        data_mesh = ax.pcolormesh(
            X_EDGES,
            Y_EDGES,
            hist.T,
            norm=data_norm,
        )

        ax.set(
            ylabel="|B| [nT]",
        )

        if i == 2:
            ax.set_xlabel("$R_H$ [au]")

    # Relative axes: How does the testing data differ from the evaluation, how
    # does the application data differ from testing.
    diffs = [
        np.zeros_like(histograms[0]),
        histograms[1] - histograms[0],
        histograms[2] - histograms[1],
    ]

    # We only want to compare our data where they overlap in data-space. Hence
    # we define a mask.
    # comparison_mask = (histograms[0] != 0) & (histograms[1] != 0) & (histograms[2] != 0)
    #
    # for i in range(len(diffs)):
    #     diffs[i][~comparison_mask] = np.nan

    largest_diff = np.nanmax(np.abs(np.array(diffs)))
    diff_norm = SymLogNorm(0.1, vmin=-largest_diff, vmax=largest_diff)

    for i, hist in enumerate(diffs):

        ax = axes[i, 1]

        # Difference from median hist goes here. Separate colour to differentiate 0 form no overlap in data
        diff_mesh = ax.pcolormesh(
            X_EDGES,
            Y_EDGES,
            hist.T,
            norm=diff_norm,
            cmap="RdBu_r",
        )

        if i == 2:
            ax.set_xlabel("$R_H$ [au]")

    for column_index in range(len(axes[:, 0])):
        _data_cadence = (
            datasets[column_index]["UTC"][2] - datasets[column_index]["UTC"][1]
        )

        axes[column_index, 0].set(
            ylabel=set_names[column_index]
            + "\n|B| [nT]\n"
            + f"N = {len(datasets[column_index].drop_nulls())}\n"
            + f"{_data_cadence * len(datasets[column_index])}",
        )

    panel_labels = "abcdef"
    for ax, c in zip(axes.T.flatten(), panel_labels):
        ax.text(0.02, 0.95, f"({c})", ha="left", va="top", transform=ax.transAxes)

    axes[0, 1].text(
        0.5, 0.5, "N/A", ha="center", va="center", transform=axes[0, 1].transAxes
    )
    axes[1, 1].text(
        0.5, 0.95, f"Panel (b) - panel (a)", ha="center", va="top", transform=axes[1, 1].transAxes
    )
    axes[2, 1].text(
        0.5, 0.95, f"Panel (c) - panel (b)", ha="center", va="top", transform=axes[2, 1].transAxes
    )

    fig.colorbar(data_mesh, ax=axes[:, 0], label="Normalised Observations [arb.]")
    fig.colorbar(diff_mesh, ax=axes[:, 1], label="Difference [arb.]")

    fig.suptitle(split_description)

    # Another figure to ignore heliocentric distance and just look at full
    # distribution.
    fig, ax = plt.subplots()

    hist_params = {
        "bins": Y_EDGES,
        "histtype": "step",
        "density": True,
        "lw": 3,
    }

    ax.hist(evaluation_data["|B| [nT]"], color=RED, label="Evaluation", **hist_params)
    ax.hist(testing_data["|B| [nT]"], color=BLUE, label="Testing", **hist_params)
    ax.hist(
        application_data["|B| [nT]"], color=BLACK, label="Application", **hist_params
    )

    ax.legend()

    ax.set_xlabel("|B| [nT]")
    ax.set_ylabel("Normalised Occurrence")

    plt.show()


def remove_flybys(data: pl.DataFrame) -> pl.DataFrame:

    _data_cadence = data["UTC"][1] - data["UTC"][0]
    _previous_data_length = len(data)

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
    print(
        f"Removed {( _previous_data_length - len(data) ) * _data_cadence} from flybys"
    )
    _previous_data_length = len(data)

    return data


def remove_non_solar_wind(data: pl.DataFrame, buffer: dt.timedelta) -> pl.DataFrame:

    # Download Hollman+ 2026 crossing list
    return filter_messenger_mag(data, buffer)


# Splitting functions
def split_trivially(
    data_to_split: pl.DataFrame, split_date: dt.datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splitting via a single threshold in time. Bound to be bad.
    """

    evaluation_data = data_to_split.filter(pl.col("UTC") <= split_date)
    testing_data = data_to_split.filter(pl.col("UTC") > split_date)

    return evaluation_data, testing_data


def split_sequentially(
    data_to_split: pl.DataFrame, chunk_length: dt.timedelta
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splitting evaluation and testing by separating alternate chunks of fixed
    length.
    """

    # Start from the begining, and separate out chunks.
    start_time = data_to_split["UTC"][0]

    chunked_mask = (
        (
            (pl.col("UTC") - start_time).dt.total_seconds()
            // chunk_length.total_seconds()
        )
        % 2
    ) == 0

    evaluation_data = data_to_split.filter(chunked_mask)
    testing_data = data_to_split.filter(~chunked_mask)

    return evaluation_data, testing_data


def chunk_length_stats(data: pl.DataFrame, chunk_length: dt.timedelta) -> pl.DataFrame:
    """Row-count statistics of chunks in the evaluation and testing subsets."""
    start_time = data["UTC"][0]

    chunk_sizes = (
        data.with_columns(
            (
                (pl.col("UTC") - start_time).dt.total_seconds()
                // chunk_length.total_seconds()
            ).alias("chunk")
        )
        .group_by("chunk")
        .agg(pl.len().alias("n_rows"))
        .with_columns(
            pl.when(pl.col("chunk") % 2 == 0)
            .then(pl.lit("evaluation"))
            .otherwise(pl.lit("testing"))
            .alias("subset")
        )
    )

    return (
        chunk_sizes.group_by("subset")
        .agg(
            pl.len().alias("n_chunks"),
            pl.col("n_rows").mean().alias("mean_rows"),
            pl.col("n_rows").median().alias("median_rows"),
            pl.col("n_rows").std().alias("std_rows"),
            pl.col("n_rows").min().alias("min_rows"),
            pl.col("n_rows").max().alias("max_rows"),
            (pl.col("n_rows").max() - pl.col("n_rows").min()).alias("range_rows"),
        )
        .sort("subset")
    )


if __name__ == "__main__":
    main()
