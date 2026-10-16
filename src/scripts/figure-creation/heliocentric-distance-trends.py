"""
Scatter B-field for evaluation, validation, and application data sets as a
function of heliocentric distance.
"""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.gridspec import GridSpec
from sunpy.time import TimeRange

from mvswm.data import Spacecraft, filter_messenger_mag
from mvswm.data.solar_cycle import get_sunspot_numbers
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

    data_to_split = data.filter(
        pl.col("UTC") <= MESSENGER_ORBIT_START - MISSION_START_BUFFER
    )
    application_data = data.filter(pl.col("UTC") > MESSENGER_ORBIT_START)

    # Clean data by removing MESSENGER magnetosphere times
    application_data = remove_non_solar_wind(
        application_data, buffer=dt.timedelta(minutes=10)
    )

    # Split description
    split_date = dt.datetime(2010, 1, 1)
    evaluation_data, testing_data = split_trivially(data_to_split, split_date)
    split_description = f"Splitting Evaluation and Testing on date: {split_date}"

    # chunk_length = dt.timedelta(days=88)
    # evaluation_data, testing_data = split_sequentially(
    #     data_to_split, chunk_length=chunk_length
    # )
    # split_description = (
    #     f"Splitting Evaluation and Testing sequentially in chunks of {chunk_length}"
    # )
    # stats = chunk_length_stats(data_to_split, chunk_length)
    # print(stats)

    # Plotting
    # Two columns of axes, one for the data, another to compare with the
    # average of all 3 rows. Each row is a different dataset
    fig = plt.figure(layout="constrained")
    gs = GridSpec(5, 2, figure=fig)

    # 1D Timeseries for context
    sun_distance_ax = fig.add_subplot(gs[0, :])
    sunspot_number_ax = fig.add_subplot(gs[1, :], sharex=sun_distance_ax)

    # 2D Histograms comparing datasets
    evaluation_ax = fig.add_subplot(gs[2, 0])
    testing_ax = fig.add_subplot(gs[3, 0], sharex=evaluation_ax, sharey=evaluation_ax)
    application_ax = fig.add_subplot(
        gs[4, 0], sharex=evaluation_ax, sharey=evaluation_ax
    )

    # Steal the unused evaluation delta axis for flattened distributions
    flat_distributions_ax = fig.add_subplot(gs[2, 1], sharey=evaluation_ax)

    testing_delta_ax = fig.add_subplot(
        gs[3, 1], sharex=evaluation_ax, sharey=evaluation_ax
    )
    application_delta_ax = fig.add_subplot(
        gs[4, 1], sharex=evaluation_ax, sharey=evaluation_ax
    )

    data_axes = [evaluation_ax, testing_ax, application_ax]
    data_delta_axes = [flat_distributions_ax, testing_delta_ax, application_delta_ax]

    datasets = [evaluation_data, testing_data, application_data]
    histograms = [
        # Density = True to normalise by area and allow comparison of datasets with different sizes.
        np.histogram2d(d["Radius [au]"], d["|B| [nT]"], bins=BINS, density=True)[0]
        for d in datasets
    ]

    data_norm = LogNorm(vmin=0.01, vmax=np.max(np.array(histograms)))

    # Data axes
    for i, (hist, ax) in enumerate(zip(histograms, data_axes)):

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

    for i, (hist, ax) in enumerate(zip(diffs, data_delta_axes)):

        if i == 0:
            continue

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

    for row_index, ax in enumerate(data_axes):
        _data_cadence = datasets[row_index]["UTC"][2] - datasets[row_index]["UTC"][1]

        ax.set(
            ylabel=set_names[row_index]
            + "\n|B| [nT]\n"
            + f"N = {len(datasets[row_index].drop_nulls())}\n"
            + f"{_data_cadence * len(datasets[row_index])}",
        )

    # evaluation_delta_ax.text(
    #     0.5,
    #     0.5,
    #     "N/A",
    #     ha="center",
    #     va="center",
    #     transform=evaluation_delta_ax.transAxes,
    # )
    testing_delta_ax.text(
        0.5,
        0.95,
        f"Panel (d) - panel (c)",
        ha="center",
        va="top",
        transform=testing_delta_ax.transAxes,
    )
    application_delta_ax.text(
        0.5,
        0.95,
        f"Panel (e) - panel (d)",
        ha="center",
        va="top",
        transform=application_delta_ax.transAxes,
    )

    fig.colorbar(data_mesh, ax=data_axes, label="Normalised Observations [arb.]")
    fig.colorbar(diff_mesh, ax=data_delta_axes[1:], label="Difference [arb.]")

    fig.suptitle(split_description)

    # Another figure to ignore heliocentric distance and just look at full
    # distribution.
    hist_params = {
        "bins": Y_EDGES.tolist(),
        "histtype": "step",
        "density": True,
        "lw": 3,
        "orientation": "horizontal",
    }

    flat_distributions_ax.hist(
        evaluation_data["|B| [nT]"], color=RED, label="Evaluation", **hist_params
    )
    flat_distributions_ax.hist(
        testing_data["|B| [nT]"], color=BLUE, label="Testing", **hist_params
    )
    flat_distributions_ax.hist(
        application_data["|B| [nT]"], color=BLACK, label="Application", **hist_params
    )

    flat_distributions_ax.legend()

    flat_distributions_ax.set_ylabel("|B| [nT]")
    flat_distributions_ax.set_xlabel("Normalised Occurrence")

    # Third figure to show time series of heliocentric distance and sunspot number, coloured by data set
    sunspots = get_sunspot_numbers()

    sunspots = (
        sunspots.with_columns(
            _year=pl.col("Decimal Year").floor().cast(pl.Int32),
        )
        .with_columns(
            _frac=pl.col("Decimal Year") - pl.col("_year"),
            _start=pl.datetime(pl.col("_year"), 1, 1),
            _end=pl.datetime(pl.col("_year") + 1, 1, 1),
        )
        .with_columns(
            UTC=pl.col("_start")
            + pl.duration(
                microseconds=(
                    (pl.col("_end") - pl.col("_start")).dt.total_microseconds()
                    * pl.col("_frac")
                ).cast(pl.Int64)
            )
        )
        .drop("_year", "_frac", "_start", "_end")
    )

    for data, color in [
        (evaluation_data, RED),
        (testing_data, BLUE),
        (application_data, BLACK),
    ]:
        d = with_gaps(data)

        d_sunspot = with_gaps(
            d.with_columns(pl.col("UTC").cast(pl.Datetime("us"))).join_asof(
                sunspots.with_columns(pl.col("UTC").cast(pl.Datetime("us"))),
                on="UTC",
                strategy="nearest",
            )
        )

        sun_distance_ax.plot(
            d["UTC"].to_numpy(), d["Radius [au]"].to_numpy(), color=color
        )
        sunspot_number_ax.plot(
            d_sunspot["UTC"].to_numpy(), d_sunspot["Mean"].to_numpy(), color=color
        )

    sun_distance_ax.set_ylabel("$R_H$ [au]")
    sunspot_number_ax.set_ylabel("Mean Monthly\nSunspot Number")

    # Label axes
    ax_labels = "abcdefghijk"
    for ax, label in zip(
        [sun_distance_ax, sunspot_number_ax, *data_axes, *data_delta_axes], ax_labels
    ):
        ax.text(0.01, 0.95, f"({label})", ha="left", va="top", transform=ax.transAxes)

    plt.show()


def with_gaps(
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
