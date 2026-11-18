"""
Plot outputs from test-training-model.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from xarray import DataArray, Dataset

from mvswm.utils.colours import BLACK, BLUE, GREEN, PINK, RED


def main() -> None:

    check_arguments()

    # The inputs and outputs of each model run are saved to a netcdf file with
    # a group for each chunk.
    model_outputs: list[Dataset] = load_model_outputs(path=Path(sys.argv[1]))

    print(
        f"Loaded {len(model_outputs)} chunks. Min index: 0, max index: {len(model_outputs) - 1}"
    )
    slice_min = int(input("Start slice at: "))
    slice_max = int(input("End slice at: "))

    datasets_to_plot = model_outputs[slice_min : slice_max + 1]

    fig, axes = plt.subplots(
        5, 1, height_ratios=[1, 1, 1, 1, 0.5], sharex=True, layout="constrained"
    )

    mag_axes = axes[:4]
    distance_ax = axes[4]

    for dataset in datasets_to_plot:
        plot_data(dataset, mag_axes)
        plot_model(dataset, mag_axes)
        plot_linear_interpolation(mag_axes)
        plot_heliocentric_distance(dataset, distance_ax)

    # Ensures no duplicate legend entries
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        ncol=5,
    )

    # Set tick format
    loc = AutoDateLocator()
    axes[0].xaxis.set_major_locator(loc)
    axes[0].xaxis.set_major_formatter(ConciseDateFormatter(loc))

    # No margins
    for ax in axes:
        ax.margins(x=0)

    plt.show()


def load_model_outputs(path: Path) -> list[Dataset]:

    model_groups = xr.open_groups(path)

    model_outputs: list[Dataset] = []
    for chunk_id, data in model_groups.items():
        model_outputs.append(data)

    # First chunk is always empty:
    model_outputs = model_outputs[1:]

    return model_outputs


def check_arguments() -> None:
    # First is the file name, second is the data path.
    if len(sys.argv) != 2:
        raise ValueError(
            "Unknown commmand. \n\nUSAGE: python plot-model-outputs.py data-file.nc"
        )


# Plotting functions
def plot_data(data: Dataset, axes: list[Axes]) -> None:

    components = ["|B| [nT]", "Br [nT]", "Bt [nT]", "Bn [nT]"]
    component_colours = [BLACK, RED, GREEN, BLUE]

    for ax, component, colour in zip(axes, components, component_colours):

        # ax.plot(
        #     *nulls_as_gaps(data[component + " Train"], dim="UTC Train"),
        #     color=colour,
        #     label="Training Data",
        # )
        ax.scatter(
            data.coords["UTC Train"],
            data[component + " Train"],
            marker=".",
            color=colour,
            label="Training Data",
        )

        ax.scatter(
            data.coords["UTC Test"],
            data[component + " Test"],
            color=colour,
            marker=".",
            alpha=0.1,
            label="Testing Data",
        )

        ax.set_ylabel(component)

    # Set axis limits to range of data + some buffer
    y_buffer = 20  # nT
    data_range = max(max(data["|B| [nT] Train"]), max(data["|B| [nT] Test"]))

    axes[0].set_ylim(0, data_range + y_buffer)
    for ax in axes[1:]:
        ax.set_ylim(-(data_range) / 2 - y_buffer, (data_range) / 2 + y_buffer)

        ax.axhline(0, color="grey", ls="dotted")


def plot_model(data: Dataset, axes: list[Axes]) -> None:

    components = ["|B| [nT]", "Br [nT]", "Bt [nT]", "Bn [nT]"]
    component_colours = [BLACK, RED, GREEN, BLUE]

    for (i, ax), component, colour in zip(
        enumerate(axes), components, component_colours
    ):

        ax.plot(
            data.coords["UTC Model"],
            data[component + " Model"],
            color=PINK,
            lw=1,
            alpha=1,
            zorder=10,
            label="GPR Mean",
        )

        lower = data[component + " Model"] - 1.95 * np.sqrt(data[f"Var({component + ' Model'})"])
        upper = data[component + " Model"] + 1.95 * np.sqrt(data[f"Var({component + ' Model'})"])

        ax.fill_between(
            data.coords["UTC Model"],
            lower,
            upper,
            color=PINK,
            alpha=0.2,
            zorder=10,
            label="GPR 95%",
        )


def plot_linear_interpolation(axes: list[Axes]) -> None:
    pass


def plot_heliocentric_distance(data: Dataset, ax: Axes) -> None:

    ax.plot(data.coords["UTC Model"], data["R Model"], color="black", lw=0.5)

    ax.set_ylabel("MESSENGER-Sun\nDistance [au]")


def nulls_as_gaps(da: DataArray, dim: str, tol: float = 1) -> tuple:
    """
    Mark jumps larger than the data cadence with NaN.

    Returns
    -------
    (coords, out) : coords is the new index along `dim` (a pandas Index),
                    out is the DataArray with NaNs marking the gaps.

    insert=True  : add one NaN sample inside each gap (output is longer by
                   the number of gaps; no real data is altered).
    insert=False : keep the same length by overwriting the first sample after
                   each gap with NaN (that real value is lost).
    tol          : a step is a gap if larger than tol * cadence (median step).
    """
    da = da.sortby(dim)
    idx = da.indexes[dim]
    if len(idx) < 3:
        return idx, da

    steps = idx.to_series().diff().iloc[1:]
    cadence = steps.median()
    is_gap = steps.values > tol * cadence

    gap_fill = idx[:-1][is_gap] + cadence
    out = da.reindex({dim: idx.union(gap_fill)})

    return out.indexes[dim], out


if __name__ == "__main__":
    main()
