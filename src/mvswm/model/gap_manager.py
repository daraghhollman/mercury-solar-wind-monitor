import datetime as dt
from dataclasses import dataclass
from typing import List, Literal

import polars as pl


@dataclass
class DataGap:
    start: dt.datetime
    end: dt.datetime


class GapManager:

    def __init__(
        self,
        data: pl.DataFrame,
        gap_length: dt.timedelta,
        strategy: Literal["middle"] = "middle",
    ) -> None:

        self._data = data
        self.strategy = strategy
        self.gap_length = gap_length

    @staticmethod
    def _overlaps(a: DataGap, b: DataGap) -> bool:
        """True if gap `a` and gap `b` overlap at all (inclusive bounds)."""
        return a.start <= b.end and b.start <= a.end

    @property
    def real_gaps(self) -> List[DataGap]:

        # Gaps are marked by nan values in the data.
        nan_rows = self._data.filter(pl.any_horizontal(pl.all().is_null()))

        # Find time differences
        nan_rows = nan_rows.with_columns(
            (pl.col("UTC") - pl.col("UTC").shift()).alias("Time Difference")
        )

        splits: List[pl.DataFrame] = nan_rows.with_columns(
            (pl.col("Time Difference") > pl.col("Time Difference").median())
            .cum_sum()
            .alias("Gap Id")
        ).partition_by("Gap Id", maintain_order=True)

        # Due to the nature of this shift time-difference, the first row
        # doesn't have a Gap Id, and needs to be merged.
        splits = [pl.merge_sorted(splits[:2], "UTC")] + splits[2:]

        # Data gaps are just the start and end times of these splits
        gaps: List[DataGap] = []
        for split in splits:

            gaps.append(DataGap(split["UTC"][0], split["UTC"][-1]))

        return gaps

    @property
    def artificial_gaps(self) -> List[DataGap]:

        # Method of placing gaps is managed by the `strategy` parameter.
        match self.strategy:

            # Place the artificial gaps in the middle of real gaps.
            case "middle":

                real_gaps = self.real_gaps

                middle_times: List[dt.datetime] = [
                    real_gaps[i].end + (real_gaps[i + 1].start - real_gaps[i].end) / 2
                    for i in range(len(self.real_gaps) - 1)
                ]

                gaps: List[DataGap] = []
                for t in middle_times:
                    gaps.append(
                        DataGap(t - self.gap_length / 2, t + self.gap_length / 2)
                    )

            case _:
                raise ValueError(
                    f"Unknown GapManager placement strategy: '{self.strategy}'"
                )

        # Remove any artificial gaps that overlap with a real gap.
        real_gaps = self.real_gaps
        gaps = [
            gap
            for gap in gaps
            if not any(self._overlaps(gap, real_gap) for real_gap in real_gaps)
        ]

        return gaps

    @property
    def training_data(self) -> pl.DataFrame | None:
        """
        Training data consists of the input dataset, minus the artificial gaps.
        """

        data = self._data

        subsets: List[pl.DataFrame] = []
        for gap in self.artificial_gaps:
            subsets.append(data.filter(~pl.col("UTC").is_between(gap.start, gap.end)))

        return pl.concat(subsets) if len(subsets) != 0 else None

    @property
    def evaluation_data(self) -> pl.DataFrame | None:
        """
        Evaluation data consists of the artificial gaps from the input data.
        """

        data = self._data

        subsets: List[pl.DataFrame] = []
        for gap in self.artificial_gaps:
            subsets.append(data.filter(pl.col("UTC").is_between(gap.start, gap.end)))

        return pl.concat(subsets) if len(subsets) != 0 else None
