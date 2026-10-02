"""
Scatter B-field for evaluation, validation, and application data sets as a
function of heliocentric distance.
"""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LogNorm
from sunpy.time import TimeRange

from mvswm.data import Spacecraft

# TODO: needs refining
MESSENGER_ORBIT_START = dt.datetime(2011, 3, 1)

X_EDGES = np.linspace(0.25, 1, 100)
Y_EDGES = np.linspace(0, 100, 1000)
BINS = [X_EDGES, Y_EDGES]


def main() -> None:

    data = Spacecraft("MESSENGER").data

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

    # Clean data by removing MESSENGER magnetosphere times

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

    data_to_split = data.filter(pl.col("UTC") <= MESSENGER_ORBIT_START)
    application_data = data.filter(pl.col("UTC") > MESSENGER_ORBIT_START)

    evaluation_data, testing_data = split_trivially(
        data_to_split, dt.datetime(2010, 1, 1)
    )

    # Plotting
    # Two columns of axes, one for the data, another to compare with the
    # average of all 3 rows. Each row is a different dataset
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)

    datasets = [evaluation_data, testing_data, application_data]
    histograms = [
        np.histogram2d(d["Radius [au]"], d["|B| [nT]"], bins=BINS)[0] for d in datasets
    ]
    median_hist = np.median(histograms, axis=0)

    # Normalise data
    vmax = max(h.max() for h in histograms)
    norm_data = LogNorm(vmin=1, vmax=vmax)

    # Data axes
    for i, hist in enumerate(histograms):

        ax = axes[i, 0]

        mesh = ax.pcolormesh(
            X_EDGES,
            Y_EDGES,
            hist.T,
            norm=norm_data,
        )

        ax.set(
            ylabel="|B| [nT]",
        )

        if i == 2:
            ax.set_xlabel("$R_H$ [au]")

    # Relative axes
    diffs = [h - median_hist for h in histograms]
    for i, hist in enumerate(diffs):

        ax = axes[i, 1]

        # Difference from median hist goes here. Separate colour to differentiate 0 form no overlap in data
        mesh = ax.pcolormesh(
            X_EDGES,
            Y_EDGES,
            hist.T,
        )

        ax.set(
            ylabel="|B| [nT]",
        )

        if i == 2:
            ax.set_xlabel("$R_H$ [au]")

    plt.show()


def split_trivially(
    data_to_split: pl.DataFrame, split_date: dt.datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splitting via a single threshold in time. Bound to be bad.
    """

    evaluation_data = data_to_split.filter(pl.col("UTC") <= split_date)
    testing_data = data_to_split.filter(pl.col("UTC") > split_date)

    return evaluation_data, testing_data


if __name__ == "__main__":
    main()
