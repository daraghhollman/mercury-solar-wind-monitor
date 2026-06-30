import datetime as dt
from pathlib import Path
from typing import List, Sequence, overload

import numpy as np
import polars as pl
import requests
import scipy.signal

type TimeOrTimes = dt.datetime | Sequence[dt.datetime]


@overload
def get_solar_cycle_phase(t: dt.datetime) -> float: ...
@overload
def get_solar_cycle_phase(t: Sequence[dt.datetime]) -> List[float]: ...


def get_solar_cycle_phase(
    t: TimeOrTimes, cache_dir=Path(__file__).parents[3] / "data/cache/"
) -> float | List[float]:

    cache_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(t, List):
        input_is_vector = True
        input_times = t

    elif isinstance(t, dt.datetime):
        input_is_vector = False
        # For ease of writing, we consider the single value input as a list of length 1.
        input_times = [t]

    else:
        raise ValueError(
            "Invalid input to `get_solar_cycle_phase`, must be time or list of times"
        )

    # Convert to decimal year
    decimal_years = [datetime_to_decimal_year(t) for t in input_times]

    # Fetch sunnspot numbers
    sunspot_numbers = (
        get_sunspot_numbers(cache_dir=cache_dir)
        .select("Decimal Year", "Mean")
        .filter(pl.col("Decimal Year") > 1960)
    )

    # Peaks should be ~11 years appart. A minmum peak distance of 6 years
    # visually works well.
    maxima_indices, _ = scipy.signal.find_peaks(
        sunspot_numbers["Mean"], distance=6 * 12
    )
    minima_indices, _ = scipy.signal.find_peaks(
        -sunspot_numbers["Mean"], distance=6 * 12
    )
    maxima = sunspot_numbers["Decimal Year"][maxima_indices]
    minima = sunspot_numbers["Decimal Year"][minima_indices]

    # Combine maxima and minima and sort
    reference_times = np.concatenate([maxima, minima])
    labels = np.concatenate([np.zeros(len(maxima)), np.full(len(minima), np.pi)])

    sort_order = np.argsort(reference_times)
    reference_times = reference_times[sort_order]
    labels = labels[sort_order]

    offset = labels[0]  # 0 if first is a maximum, π if first is a minimum
    known_phases = np.arange(len(labels)) * np.pi + offset

    # Find phase at time
    ouput_phases = [
        t % (2 * np.pi)
        for t in np.interp(
            decimal_years,
            reference_times,
            known_phases,
        ).ravel()
    ]

    # _, ax = plt.subplots()
    # ax.plot(sunspot_numbers["Decimal Year"], sunspot_numbers["Mean"])
    # for m in maxima:
    #     ax.axvline(m, color="red")
    #
    # for m in minima:
    #     ax.axvline(m, color="green")
    #
    # ax.plot(decimal_years, ouput_phases)
    #
    # plt.show()

    return ouput_phases if input_is_vector else ouput_phases[0]


def get_sunspot_numbers(
    cache_dir: Path = Path(__file__).parents[3] / "data/cache/",
) -> pl.DataFrame:
    """Fetches a timeseries of sunspot number"""

    # 13 month smoothed total sunspot number
    url = "https://www.sidc.be/SILSO/INFO/snmstotcsv.php"
    path = cache_dir / "monthly_sunspot_number.csv"

    # Ensure cache_dir exists
    cache_dir.mkdir(exist_ok=True)

    # Check if the file exists before downloading:
    if not path.exists():
        response = requests.get(url)
        with open(path, "wb") as file:
            file.write(response.content)

    data = pl.read_csv(
        path,
        new_columns=[
            "Year",
            "Month",
            "Decimal Year",
            "Mean",
            "Standard Deviation",
            "N Observations",
            "Provisional Marker",
        ],
        schema_overrides={
            "Mean": pl.Float64,
            "Standard Deviation": pl.Float64,
            "N Observations": pl.Float64,
        },
        has_header=False,
        separator=";",
    )

    return data


def datetime_to_decimal_year(d: dt.datetime) -> float:
    year_start = dt.datetime(d.year, 1, 1)
    year_end = dt.datetime(d.year + 1, 1, 1)
    year_length = (year_end - year_start).total_seconds()

    elapsed = (d - year_start).total_seconds()

    return d.year + elapsed / year_length
