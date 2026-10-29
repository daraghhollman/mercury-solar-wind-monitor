"""
Scatter |B| against sunspot number for MESSENGER orbit phase, and cruise within
Mercury heliocentric distances to better understand the trend in solar cycle.
"""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sunpy.time import TimeRange

from mvswm.data import Spacecraft, filter_messenger_mag, get_icmes
from mvswm.data.solar_cycle import get_sunspot_numbers

# The magnetosphere removal function
MESSENGER_ORBIT_START = dt.datetime(2011, 3, 21)

SUNSPOT_BINS = np.linspace(0, 120, 20)
MAGNITUDE_BINS = np.linspace(0, 100, 100)


def main() -> None:

    data = Spacecraft("MESSENGER").data

    # Clean data by removing flybys of Earth, Venus, and Mercury
    data = remove_flybys(data)

    # Clean data by removing anything outside Mercury heliocentric distances
    data = data.filter(pl.col("Radius [au]") < 0.48)

    # Clean data by removing icmes
    data = remove_icmes(data)

    pre_orbital_phase = data.filter(pl.col("UTC") <= MESSENGER_ORBIT_START)
    orbital_phase = data.filter(pl.col("UTC") > MESSENGER_ORBIT_START)

    orbital_phase = remove_non_solar_wind(
        orbital_phase, buffer=dt.timedelta(minutes=10)
    ).with_columns(pl.col("UTC").cast(pl.Datetime("ns")))

    data = pl.concat([pre_orbital_phase, orbital_phase])

    sunspots = get_sunspot_numbers()

    data = data.join_asof(
        sunspots, left_on="UTC", right_on="Month Start", strategy="backward"
    ).rename({"Mean": "Mean Sunspots"})

    hist, _, _ = np.histogram2d(
        data["Mean Sunspots"], data["|B| [nT]"], bins=(SUNSPOT_BINS, MAGNITUDE_BINS)
    )

    # Fitting
    fit_data = data.drop_nulls()
    fit_pars, fit_cov = curve_fit(
        linear_fit, fit_data["Mean Sunspots"], fit_data["|B| [nT]"]
    )

    # Pearson R
    correlation = pearsonr(fit_data["Mean Sunspots"], fit_data["|B| [nT]"])
    print(correlation)

    fig, ax = plt.subplots()

    ax.pcolormesh(
        SUNSPOT_BINS,
        MAGNITUDE_BINS,
        hist.T,
        norm=LogNorm(),
    )

    fit_sigma = np.sqrt(np.diag(fit_cov))

    fit_label = (
        f"y = mx + c"
        + f"\n m = {fit_pars[0]:.3f} +/- {fit_sigma[0]:.3f}"
        + f"\n  c = {fit_pars[1]:.1f} +/- {fit_sigma[1]:.1f}"
    )

    _bin_centres = (SUNSPOT_BINS[1:] + SUNSPOT_BINS[:-1]) / 2
    ax.plot(
        _bin_centres,
        linear_fit(_bin_centres, *fit_pars),
        color="black",
        label=fit_label,
    )

    ax.set(
        xlabel="Mean Sunspot Number (Monthly)",
        ylabel="|B| [nT]",
    )

    ax.legend()

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
    return filter_messenger_mag(data, buffer)


def remove_icmes(data: pl.DataFrame) -> pl.DataFrame:
    icmes = get_icmes()

    # Filter to only MESSENGER
    icmes = icmes.filter(pl.col("sc_insitu") == "MESSENGER")

    data = data.filter(
        ~pl.any_horizontal(
            pl.col("UTC").is_between(start, end)
            for start, end in icmes.select(
                "icme_start_time", "mo_end_time"
            ).iter_rows()
        )
    )

    return data


def linear_fit(x, m, c):
    return m * x + c


if __name__ == "__main__":
    main()
