"""
To be certain in our method for determining metrics, we show simple examples
for test cases: linear, sin, ...
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from gpflow.kernels import RationalQuadratic
from gpflow.models import GPR
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from scipy.stats import linregress

from mvswm.utils.colours import GREEN, ORANGE, RED

ADD_GPR = False


def main() -> None:

    tests: List[str] = ["linear", "discontinuity", "sin", "noisy sin"]

    fig, axes = plt.subplots(
        len(tests), 3, width_ratios=[4, 2, 1], figsize=(10, 2 * len(tests))
    )

    for i, which in enumerate(tests):

        y_data_original = get_data(which)
        x_data = np.arange(len(y_data_original))

        # Add gap based on indices
        gap_indices = (40, 60)
        gap_width = gap_indices[1] - gap_indices[0]

        x_test = range(*gap_indices)

        y_data = y_data_original.copy()
        y_test = y_data[gap_indices[0] : gap_indices[1]].copy()

        # Set the gap data to nan
        y_data[gap_indices[0] : gap_indices[1]] = np.nan

        # Linearly interpolate accross the gap. We mask the nans to ignore the gap
        mask = ~np.isnan(y_data)
        y_li = np.interp(x_test, x_data[mask], y_data[mask])

        # Add GPR predictions
        if ADD_GPR:
            model = GPR(
                (
                    x_data[mask].astype(np.float64).reshape(-1, 1),
                    y_data[mask].astype(np.float64).reshape(-1, 1),
                ),
                kernel=RationalQuadratic(),
                noise_variance=1e-4,
            )
            y_gpr, y_gpr_var = model.predict_y(
                np.array(x_test).astype(np.float64).reshape(-1, 1)
            )

            y_gpr = y_gpr.numpy().flatten()
            y_gpr_var = y_gpr_var.numpy().flatten()

        # Plotting

        # First panel, data with gap
        ax = axes[i, 0]
        ax: Axes

        ax.plot(x_data, y_data, color="black", label="Training Data")

        ax.scatter(
            x_test,
            y_test,
            color="black",
            marker="o",
            alpha=0.2,
            label="Evaluation Data",
        )

        ax.scatter(x_test, y_li, color=ORANGE, marker=".", label="LI")

        if ADD_GPR:
            ax.errorbar(
                x_test,
                y_gpr,
                fmt=".",
                yerr=np.sqrt(y_gpr_var),
                color=GREEN,
                label="GPR",
            )

        if i == 0:
            ax.legend()

        ax.set_xlabel("x [arb.]")
        ax.set_ylabel("y [arb.]")

        ax.margins(x=0)

        # Second panel, observed vs predicted
        ax = axes[i, 1]
        ax: Axes

        # Get correlation
        correlation = linregress(y_test, y_li).rvalue

        ax.scatter(
            y_test, y_li, color="black", marker=".", label=f"$r_p=$ {correlation:.2f}"
        )

        min_value = np.min([y_test, y_li])
        max_value = np.max([y_test, y_li])

        # 45 degree line
        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            color=RED,
            ls="dashed",
            zorder=-1,
        )

        ax.set_aspect("equal")
        ax.margins(0)

        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

        ax.legend()

        # 3rd panel: What distribution of correlation do we get by sliding the gap from left to right

        ax = axes[i, 2]
        ax: Axes

        slide_correlations = []
        starts = range(0, len(y_data_original) - gap_width + 1)
        for start in starts:
            end = start + gap_width

            y_slide = y_data_original.copy()
            x_test_slide = range(start, end)
            y_test_slide = y_slide[start:end].copy()
            y_slide[start:end] = np.nan

            mask_slide = ~np.isnan(y_slide)
            y_li_slide = np.interp(
                x_test_slide, x_data[mask_slide], y_slide[mask_slide]
            )

            slide_correlations.append(linregress(y_test_slide, y_li_slide).rvalue)

        # Remove first and last to avoid edge cases
        slide_correlations = slide_correlations[1:-1]

        ax.hist(
            slide_correlations,
            orientation="horizontal",
            bins=np.arange(-1, 1 + 0.1, 0.1).tolist(),
            color="black",
        )

        ax.set_ylabel("$r_p$")
        ax.set_xticks([])

        ax.margins(y=0)

        # Add a boxplot for this distribution
        divider = make_axes_locatable(ax)
        ax_box: Axes = divider.append_axes("right", size="20%", pad=0, sharey=ax)

        ax_box.boxplot(
            slide_correlations,
            vert=True,
            widths=0.6,
            medianprops=dict(color="black"),
            flierprops=dict(clip_on=False),
        )
        ax_box.set_xticks([])
        ax_box.set_ylim(ax.get_ylim())  # keep y-axis aligned with the histogram

        # Hide the boxplot's own y-axis labels since it shares ax's
        ax_box.tick_params(left=False, labelleft=False)

        ax_box.set_frame_on(False)

    fig.suptitle(f"GAP SIZE: {gap_width}")

    fig.savefig(f"./figures/correlation-tests.pdf", format="pdf", bbox_inches="tight")


def get_data(which: str) -> NDArray[np.float64]:

    data_range = np.linspace(0, 1, 100)

    match which.lower():

        case "linear":
            return data_range

        case "discontinuity":
            discontinuity = data_range
            discontinuity[:len(data_range) // 2] -= 1

            return discontinuity

        case "sin":
            return np.sin(data_range * 15)

        case "noisy sin":
            return np.sin(data_range * 15) + np.random.normal(
                scale=0.3, size=len(data_range)
            )

        case _:
            raise ValueError(f"No matching data function for `which`=={which}")


if __name__ == "__main__":
    main()
