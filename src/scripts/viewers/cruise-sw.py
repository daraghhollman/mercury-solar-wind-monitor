"""
A script to view mutliple orbits of solar wind data from MESSENGER cruise. This
script uses a local cache file to record what was last looked at.

We isolate only data from August 2007 and onwards to look at times in
near-Mercury heliocentric distances.
"""

import datetime as dt
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import requests
import spiceypy as spice
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.table import QTable
from astropy.time import Time, TimeDelta
from hermpy.data import add_field_magnitude, parse_messenger_mag
from hermpy.net import ClientMESSENGER, ClientSPICE
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from sunpy.coordinates.frames import HeliographicStonyhurst
from sunpy.time import TimeRange

INDEX_CACHE_FILE = Path(__file__).parent / ".cruise-index.cache"
ICME_CACHE_FILE = Path(__file__).parent / ".icme.cache"
FULL_TIME_RANGE = TimeRange("2008-04-01", "2011-03-01")

# It still makes sense to visualise these data on the scale of one orbit
PANEL_LENGTH = dt.timedelta(days=27)
N_PANELS: int = 4


def main():

    icmes = get_icmes()

    while True:
        # Get the current index
        i = get_current_panel_index()

        first_panel_start = FULL_TIME_RANGE.start.to_datetime() + i * PANEL_LENGTH
        first_panel_end = FULL_TIME_RANGE.start.to_datetime() + (i + 1) * PANEL_LENGTH

        if first_panel_end > FULL_TIME_RANGE.end:
            print("No more cruise data")
            return

        # Load data for this index
        data_time_range = TimeRange(
            first_panel_start,
            first_panel_start + N_PANELS * PANEL_LENGTH,
        )

        messenger = ClientMESSENGER()
        messenger.query(
            data_time_range,
            instrument="MAG RTN 60s",
        )

        # Download data for this and surrounding orbits
        data = parse_messenger_mag(messenger.fetch(), data_time_range)
        data = add_field_magnitude(data)

        spice_client = ClientSPICE()
        spice_client.KERNEL_LOCATIONS.update(
            {
                "MESSENGER": {
                    "BASE": "https://naif.jpl.nasa.gov/pub/naif/",
                    "DIRECTORY": "pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/spk/",
                    "PATTERNS": ["msgr_??????_??????_??????_od431sc_2.bsp"],
                },
            }
        )

        with spice_client.KernelPool():
            create_plot(data, icmes, i)

        input("Press ENTER to continue")
        set_current_orbit_index(i + N_PANELS)


def create_plot(data: QTable, icmes: QTable, orbit_index: int) -> None:

    fig = plt.figure()
    gs = GridSpec(N_PANELS, 2, figure=fig, width_ratios=[1.5, 1], wspace=0.1)

    axes = [fig.add_subplot(gs[i, 0]) for i in range(N_PANELS)]
    trajectory_ax = fig.add_subplot(gs[:-1, 1], projection="polar")

    window_start = FULL_TIME_RANGE.start.to_datetime() + orbit_index * PANEL_LENGTH
    window_start_time = Time(window_start, scale="utc")
    panel_days = PANEL_LENGTH.total_seconds() / 86400  # 27.0

    # Days since the start of the whole window, for every sample
    if len(data) > 0:
        t = np.atleast_1d((data["UTC"] - window_start_time).to_value("day"))
    else:
        t = np.array([])

    # Split by time, not by number of rows
    panel_masks = [
        (t >= i * panel_days) & (t < (i + 1) * panel_days) for i in range(N_PANELS)
    ]
    all_panels_data = [data[m] for m in panel_masks]

    # Elapsed days within each panel, anchored to the panel's nominal start
    elapsed = [t[m] - i * panel_days for i, m in enumerate(panel_masks)]

    # Convert icme times to panel coordinates
    if icmes is not None and len(icmes) > 0:
        icme_start_t = np.atleast_1d(
            (icmes["icme_start_time"] - window_start_time).to_value("day")
        )
        icme_end_t = np.atleast_1d(
            (icmes["icme_end_time"] - window_start_time).to_value("day")
        )
    else:
        icme_start_t = np.array([])
        icme_end_t = np.array([])

    trajectory_labelled = False
    panel_end_labelled = False

    for i, (ax, panel_data, panel_elapsed) in enumerate(
        zip(axes, all_panels_data, elapsed)
    ):
        has_data = len(panel_data) > 0

        for trace, trace_elapsed in zip(all_panels_data, elapsed):
            if len(trace) == 0:
                continue
            ax.plot(
                trace_elapsed, trace["|B|"], color="grey", lw=0.3, alpha=0.3, zorder=1
            )

        if has_data:
            ax.plot(panel_elapsed, panel_data["|B|"], color="black", lw=0.5, zorder=5)
        else:
            ax.text(
                0.5, 0.5, "No data",
                color="grey", fontsize=10, ha="center", va="center",
                transform=ax.transAxes,
            )

        # If any ICMEs happened, shade them
        # Nominal panel bounds in window-relative days
        panel_lo = i * panel_days
        panel_hi = (i + 1) * panel_days

        # If any ICMEs happened, shade them (in this panel's elapsed-day coordinates)
        in_panel = (icme_end_t > panel_lo) & (icme_start_t < panel_hi)
        for start, end in zip(icme_start_t[in_panel], icme_end_t[in_panel]):
            # Shift into the panel's frame, clipping ICMEs that straddle a boundary
            x0 = max(start, panel_lo) - panel_lo
            x1 = min(end, panel_hi) - panel_lo

            ax.axvspan(x0, x1, color="indianred", alpha=0.4)

        ax.set_xlim(0, panel_days)  # always 0-27
        ax.set_ylim(1, 100)
        ax.set_yscale("log")

        if ax == axes[-1]:
            ax.set_xlabel("Days")

        ax.xaxis.set_major_locator(MultipleLocator(3))

        if ax != axes[0]:
            ax.tick_params(axis="x", top=True, direction="inout", length=10)

        if ax != axes[-1]:
            ax.xaxis.set_ticklabels([])

        ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels()[:-1])

        # Get MESSENGER position parameters
        panel_start = window_start + i * PANEL_LENGTH

        ax.text(
            0.01,
            0.9,
            f"#{orbit_index + i}: {panel_start}",
            color="grey",
            fontsize=8,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

        if not has_data:
            continue

        ets = np.atleast_1d(spice.datetime2et(panel_data["UTC"].to_datetime()))
        if ets.size == 0:
            continue

        positions, _ = spice.spkpos("MESSENGER", ets, "J2000", "NONE", "SUN")
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        if positions.size == 0:
            continue

        positions = positions * u.km

        skycoord = SkyCoord(
            CartesianRepresentation(positions.T),
            obstime=panel_data["UTC"],
            frame="icrs",
        ).transform_to(HeliographicStonyhurst)

        assert isinstance(skycoord.radius, u.Quantity)
        assert isinstance(skycoord.lon, u.Quantity)
        assert isinstance(skycoord.lat, u.Quantity)

        lon = np.atleast_1d(skycoord.lon.to("rad").value)
        radius = np.atleast_1d(skycoord.radius.to("au").value)
        if lon.size == 0:
            continue

        trajectory_ax.plot(
            lon,
            radius,
            color="black",
            label="MESSENGER trajectory" if not trajectory_labelled else "",
        )
        trajectory_ax.scatter(
            lon[-1],
            radius[-1],
            color="indianred",
            zorder=5,
            label="Panel end" if not panel_end_labelled else "",
        )
        trajectory_labelled = True
        panel_end_labelled = True

    # Static trajectory-axis elements, drawn once regardless of data
    trajectory_ax.scatter(0, 0, color="orange", zorder=10, s=30, label="Sun")
    trajectory_ax.axhspan(
        0.3, 0.47, color="grey", alpha=0.3, label="Mercury $R_H$ extent"
    )
    trajectory_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))
    trajectory_ax.set_theta_zero_location("S")
    trajectory_ax.set_xticklabels([])

    fig.suptitle("MESSENGER MAG (Cruise)")
    fig.subplots_adjust(right=0.99, top=0.92, hspace=0)
    fig.supylabel("|B| [nT]")

    plt.savefig(Path(__file__).parent / ".cruise-sw-fig.pdf", format="pdf")


def set_current_orbit_index(index, cache_file: Path = INDEX_CACHE_FILE) -> None:
    with open(cache_file, "w") as f:
        f.write(str(index))


def get_current_panel_index(cache_file: Path = INDEX_CACHE_FILE) -> int:

    # If the file doesn't exist, create it and return 0
    if not cache_file.is_file():
        set_current_orbit_index(0)
        return get_current_panel_index()

    with open(cache_file, "r") as f:
        return int(f.read().strip())


def get_icmes(cache_file: Path = ICME_CACHE_FILE) -> QTable:

    url = "https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v23.csv"

    # If the file doesn't exist, download it
    if not cache_file.exists():
        response = requests.get(url)
        with open(cache_file, "wb") as file:
            file.write(response.content)

    icme_table: QTable = QTable.read(cache_file, format="ascii.csv")

    icme_table["icme_start_time"] = Time(icme_table["icme_start_time"])
    icme_table["icme_end_time"] = Time(icme_table["icme_start_time"]) + TimeDelta(
        icme_table["icme_duration"]
    )

    # Limit only to MESSENGER
    icme_table = icme_table[icme_table["sc_insitu"] == "MESSENGER"]

    return icme_table


if __name__ == "__main__":
    main()
