"""
A pictoral example of how Gaussian processes fit to data. Inspired by Figure
1.1 from C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine
Learning, the MIT Press, 2006, ISBN 026218253X.
"""

from matplotlib.axes import Axes
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mvswm.utils.colours import *

SEED = 0

np.random.seed(SEED)
tf.random.set_seed(SEED)

N_SAMPLES = 4
X_MIN, X_MAX = -5, 5

SAMPLE_LENGTH = 200
x = np.linspace(X_MIN, X_MAX, SAMPLE_LENGTH).reshape(-1, 1)

KERNEL = gpflow.kernels.SquaredExponential()

NOISE = 1e-4


def main() -> None:

    fig, axes = plt.subplots(1, 2, figsize=(6, 2), sharey=True)

    # LEFT PANEL - PRIOR DISTRIBUTION
    #
    #################################
    prior_covariance = KERNEL(x)
    prior_mean = np.zeros(SAMPLE_LENGTH)
    prior_sd = np.sqrt(np.diag(prior_covariance))

    # Samples from the prior
    prior_samples = np.random.multivariate_normal(
        mean=prior_mean, cov=prior_covariance, size=N_SAMPLES
    ).T

    ax = axes[0]
    ax.set_title("Prior Distribution")

    ax.plot(x, prior_samples, color="black", lw=0.5, ls="dashed")
    ax.plot(x, prior_mean, color="black", lw=2, label="Mean")

    ax.fill_between(
        x.flatten(),
        prior_mean - 1.96 * prior_sd,
        prior_mean + 1.96 * prior_sd,
        color="grey",
        alpha=0.2,
        label="95% CI",
    )

    ax.legend()

    ax.set_ylabel("Y [arb.]")

    # RIGHT PANEL - POSTERIOR DISTRIBUTION
    #
    ######################################

    # Define some observed data
    x_observed = np.array([-3.5, 0.2, 0.8, ]).reshape(-1, 1)
    y_observed = np.sin(x_observed)

    model = gpflow.models.GPR(
        (x_observed, y_observed),
        kernel=KERNEL,
        noise_variance=NOISE,
    )

    posterior_mean, posterior_variance = model.predict_f(x)

    posterior_mean = posterior_mean.numpy().flatten()
    posterior_sd = np.sqrt(posterior_variance.numpy().flatten())

    # Samples from posterior
    posterior_samples = (
        model.predict_f_samples(x, N_SAMPLES).numpy()[:, :, 0].T
    ) # (len(x), N_SAMPLES)

    ax = axes[1]
    ax.set_title("Posterior Distribution")

    ax.scatter(x_observed, y_observed, color=RED, zorder=5, label="New Obervations")

    ax.plot(x, posterior_samples, color="black", lw=0.5, ls="dashed")
    ax.plot(x, posterior_mean, color="black", lw=2)

    ax.fill_between(
        x.flatten(),
        posterior_mean - 1.96 * posterior_sd,
        posterior_mean + 1.96 * posterior_sd,
        color="grey",
        alpha=0.2,
    )

    ax.legend()

    axes_labels = "ab"
    ax: Axes
    for i, ax in enumerate(axes):
        ax.text(0.02, 0.9, f"({axes_labels[i]})", transform=ax.transAxes)
        ax.set_xlabel("X [arb.]")
        ax.set_ylim(-4, 4)
        ax.margins(x=0)

    plt.savefig("./figures/gpr-pictoral-example.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
