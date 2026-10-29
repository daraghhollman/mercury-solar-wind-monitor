from pathlib import Path

import polars as pl
import requests


def get_icmes(
    cache_dir: Path = Path(__file__).parents[3] / "data/cache/",
) -> pl.DataFrame:
    """Fetches ICMEs from ICMECAT"""

    # 13 month smoothed total sunspot number
    url = "https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v23.csv"
    path = cache_dir / "icmes.csv"

    # Ensure cache_dir exists
    cache_dir.mkdir(exist_ok=True)

    # Check if the file exists before downloading:
    if not path.exists():
        response = requests.get(url)
        with open(path, "wb") as file:
            file.write(response.content)

    data = pl.read_csv(path, try_parse_dates=True)

    data = data.with_columns(
        pl.col("icme_start_time", "mo_end_time")
        .dt.replace_time_zone(None)
        .dt.cast_time_unit("ns")
    )

    return data
