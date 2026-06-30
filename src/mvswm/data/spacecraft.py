import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

import astropy.units as u
import polars as pl
import spiceypy as spice
from astropy.coordinates import CartesianRepresentation, SkyCoord
from hermpy.net import ClientSPICE
from sunpy.coordinates.frames import HeliographicStonyhurst
from sunpy.time import TimeRange

from mvswm.data import MAG_LOADERS, get_solar_cycle_phase


@dataclass
class Spacecraft:
    name: Literal[
        "MESSENGER",
        "Solar Orbiter",
        "Parker Solar Probe",
        "Helios 1",
        "Helios 2",
    ]
    cache_dir: Path = Path(".cache/")
    ephermeris_resolution: dt.timedelta = dt.timedelta(minutes=1)

    # One polars dataframe to hold the data and positions.
    # We will cache it to not have to separate generating and loading.
    @property
    def data(self) -> pl.DataFrame:
        """
        Download or loads data from this spacecraft. Pass a TimeRange to
        filter to a specific time range.
        """

        # First check if we have the data cached
        cache_file = self.cache_dir / f"{self.name.lower().replace(' ', '-')}.parquet"

        if cache_file.exists():
            return pl.read_parquet(cache_file)

        else:
            # Download data
            print(f"Data for {self.name} not cached:")
            try:
                print("    Fetching MAG data")
                data = MAG_LOADERS[self.name].__call__()

            except KeyError as e:
                raise KeyError(f"No matching MAG loader for spacecraft '{self.name}'")

            # Fetch positions through SPICE
            print("    Fetching position data")
            start = data["UTC"][0]
            end = data["UTC"][-1]

            spice_client = ClientSPICE()
            spice_client.KERNEL_LOCATIONS.update(SPICE_KERNELS)

            with spice_client.KernelPool():

                times: Sequence[dt.datetime] = [
                    start + t * self.ephermeris_resolution
                    for t in range((end - start) // self.ephermeris_resolution)
                ]

                ets = spice.datetime2et(times)
                positions, _ = spice.spkpos(self.name, ets, "J2000", "NONE", "SUN")

            positions *= u.km

            skycoords = SkyCoord(
                CartesianRepresentation(positions.T),
                obstime=times,
                frame="icrs",
            ).transform_to(HeliographicStonyhurst)

            assert isinstance(skycoords.radius, u.Quantity)
            assert isinstance(skycoords.lon, u.Quantity)
            assert isinstance(skycoords.lat, u.Quantity)

            positions_table = pl.DataFrame(
                {
                    "UTC": times,
                    "Radius [au]": skycoords.radius.to(u.au),
                    "Longitude [deg]": skycoords.lon.to(u.deg),
                    "Latitude [deg]": skycoords.lat.to(u.deg),
                    "Solar Cycle Phase": get_solar_cycle_phase(times),
                }
            )

            # Join the positions to the MAG data by nearest neighbour
            positions_table = positions_table.with_columns(
                pl.col("UTC").cast(data.schema["UTC"])
            ).sort("UTC")

            data = data.sort("UTC").join_asof(
                positions_table,
                on="UTC",
                strategy="nearest",
            )

            # Cache to disk
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            data.write_parquet(cache_file)
            print(f"    Cached to {cache_file}")

            return data

    def get_data_in_range(self, time_range: TimeRange) -> pl.DataFrame:

        # https://github.com/sunpy/sunpy/pull/8614
        assert time_range.start is not None
        assert time_range.end is not None

        return self.data.filter(
            pl.col("UTC").is_between(
                time_range.start.to_datetime(), time_range.end.to_datetime()
            )
        )


SPICE_KERNELS: Dict[str, Dict[str, Any]] = {
    "MESSENGER": {
        "BASE": "https://naif.jpl.nasa.gov/pub/naif/",
        "DIRECTORY": "pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/spk/",
        "PATTERNS": ["msgr_??????_??????_??????_od431sc_2.bsp"],
    },
    "Solar Orbiter": {
        "BASE": "http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/",
        "DIRECTORY": "kernels/spk/",
        "PATTERNS": [
            "de421.bsp",
            "solo_ANC_soc-orbit_20200210-20301118_L000_V0_00001_V01.bsp",
        ],
    },
    "Parker Solar Probe": {
        "BASE": "https://spdf.gsfc.nasa.gov/pub/data/psp/",
        "DIRECTORY": "ephemeris/spice/ephemerides/",
        "PATTERNS": ["spp_nom_20180812_20300101_v043_PostV7.bsp"],
    },
    "Helios 1/2": {
        "BASE": "https://naif.jpl.nasa.gov/pub/naif/HELIOS/",
        "DIRECTORY": "kernels/spk/",
        "PATTERNS": [
            "???????_???????_?????_?????.bsp",
            "????????_???????_?????_?????.bsp",
        ],
    },
}
