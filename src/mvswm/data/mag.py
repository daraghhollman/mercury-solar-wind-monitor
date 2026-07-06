from functools import partial
from pathlib import Path
from typing import Callable, Dict

from sunpy.time import TimeRange

from mvswm.data.downloaders import (
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
