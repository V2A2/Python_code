import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import (
    multivariate_t,
    gaussian_kde,
    norm
)

from sklearn.mixture import GaussianMixture


# ============================================================
# Publication style for TACAS / Springer LNCS
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",

    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    "axes.linewidth": 0.8,

    # Keep fonts editable in vector output
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.bbox": "tight",
})


# ============================================================
# Configuration
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

DIM = 5
N_SAMPLES = 10_000

# Number of Gaussian components used by Generalized ProbStar
N_GMM_COMPONENTS = 10

# Predicate domain
LOWER = -np.ones(DIM)
UPPER = np.ones(DIM)


# ============================================================
# Ground-truth distribution
#
# Two-component 5D Student-t mixture
# ============================================================

true_weights = np.array([
    0.80,
    0.20
])

true_df = np.array([
    10.0,
    10.0
])

true_means = np.array([
    [-0.35, -0.30, -0.25, -0.20, -0.30],
    [ 0.55,  0.45,  0.50,  0.40,  0.50]
])


# Student-t scale matrices
scale1_diag = np.array([
    0.18,
    0.20,
    0.19,
    0.18,
    0.20
]) ** 2

scale2_diag = np.array([
    0.14,
    0.15,
    0.14,
    0.16,
    0.15
]) ** 2

true_scales = np.array([
    np.diag(scale1_diag),
    np.diag(scale2_diag)
])


# ============================================================
# Utility:
# Check whether points are inside [-1,1]^5
# ============================================================

def inside_box(x):

    x = np.atleast_2d(x)

    return np.all(
        (x >= LOWER) & (x <= UPPER),
        axis=1
    )


# ============================================================
# Sampling from the truncated true distribution
# ============================================================

def sample_true_truncated(n, rng):
    """
    Rejection sampling from the two-component Student-t mixture
    conditioned on the predicate domain [-1,1]^5.
    """

    accepted = []
    total = 0

    while total < n:

        batch_size = max(
            5000,
            2 * (n - total)
        )

        # Select Student-t component
        component_ids = rng.choice(
            2,
            size=batch_size,
            p=true_weights
        )

        batch = np.empty(
            (batch_size, DIM)
        )

        # Generate samples from each component
        for k in range(2):

            idx = np.where(
                component_ids == k
            )[0]

            if len(idx) == 0:
                continue

            batch[idx] = multivariate_t.rvs(
                loc=true_means[k],
                shape=true_scales[k],
                df=true_df[k],
                size=len(idx),
                random_state=rng
            )

        # Keep only samples inside [-1,1]^5
        mask = inside_box(batch)

        good = batch[mask]

        accepted.append(good)

        total += len(good)

    samples = np.vstack(accepted)

    return samples[:n]


# ============================================================
# 1. Generate samples
# ============================================================

print(
    "Generating samples from the truncated "
    "ground-truth distribution..."
)

samples = sample_true_truncated(
    N_SAMPLES,
    rng
)

print(
    f"Generated {samples.shape[0]} samples."
)


# ============================================================
# 2. ProbStar approximation
#
# Fit one Gaussian using all samples
# ============================================================

print(
    "Fitting single Gaussian for ProbStar..."
)

single_mean = np.mean(
    samples,
    axis=0
)

centered = (
    samples - single_mean
)

# Maximum-likelihood covariance estimate
single_cov = (
    centered.T @ centered
    / N_SAMPLES
)


# ============================================================
# 3. Generalized ProbStar approximation
#
# Fit a 10-component full-covariance GMM using EM
# ============================================================

print(
    f"Fitting {N_GMM_COMPONENTS}-component "
    "GMM for Generalized ProbStar..."
)

gmm = GaussianMixture(
    n_components=N_GMM_COMPONENTS,
    covariance_type="full",
    random_state=SEED,
    n_init=10,
    max_iter=1000,
    tol=1e-6,
    reg_covar=1e-8
)

gmm.fit(samples)

print(
    f"EM converged: {gmm.converged_}"
)

print(
    f"EM iterations: {gmm.n_iter_}"
)


# ============================================================
# Extract fitted GMM parameters
# ============================================================

gmm_weights = gmm.weights_.copy()
gmm_means = gmm.means_.copy()
gmm_covs = gmm.covariances_.copy()


# ============================================================
# Sort GMM components by weight
# ============================================================

order = np.argsort(
    -gmm_weights
)

gmm_weights = gmm_weights[order]
gmm_means = gmm_means[order]
gmm_covs = gmm_covs[order]


# ============================================================
# Evaluation grid
# ============================================================

x_grid = np.linspace(
    -1.0,
    1.0,
    1200
)


# ============================================================
# 4. Create figure
#
# 3 columns x 2 rows:
#
#   alpha_1   alpha_2   alpha_3
#   alpha_4   alpha_5   [empty]
#
# Designed for a full-width figure in TACAS / LNCS.
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(7.0, 4.3),
    sharex=True,
    sharey=True
)

axes = axes.flatten()


# ============================================================
# 5. Plot the five predicate dimensions
# ============================================================

for dim in range(DIM):

    ax = axes[dim]


    # ========================================================
    # A. Density estimated from true samples
    # ========================================================

    kde = gaussian_kde(
        samples[:, dim]
    )

    sample_density = kde(
        x_grid
    )

    # Normalize over [-1,1]
    sample_density /= np.trapezoid(
        sample_density,
        x_grid
    )


    # ========================================================
    # B. ProbStar
    #
    # One-dimensional marginal of the fitted Gaussian
    # ========================================================

    ps_sigma = np.sqrt(
        single_cov[dim, dim]
    )

    ps_density = norm.pdf(
        x_grid,
        loc=single_mean[dim],
        scale=ps_sigma
    )

    # Normalize over [-1,1]
    ps_density /= np.trapezoid(
        ps_density,
        x_grid
    )


    # ========================================================
    # C. Generalized ProbStar
    #
    # One-dimensional marginal of the fitted GMM
    # ========================================================

    gps_density = np.zeros_like(
        x_grid
    )

    for k in range(N_GMM_COMPONENTS):

        sigma_k = np.sqrt(
            gmm_covs[k, dim, dim]
        )

        gps_density += (
            gmm_weights[k]
            * norm.pdf(
                x_grid,
                loc=gmm_means[k, dim],
                scale=sigma_k
            )
        )

    # Normalize over [-1,1]
    gps_density /= np.trapezoid(
        gps_density,
        x_grid
    )


    # ========================================================
    # True Samples
    # ========================================================

    ax.plot(
        x_grid,
        sample_density,
        linewidth=1.7,
        linestyle="-",
        label="True Samples",
        zorder=3
    )


    # ========================================================
    # ProbStar
    # ========================================================

    ax.plot(
        x_grid,
        ps_density,
        linewidth=1.6,
        linestyle="--",
        label="ProbStar",
        zorder=2
    )


    # ========================================================
    # Generalized ProbStar
    # ========================================================

    ax.plot(
        x_grid,
        gps_density,
        linewidth=1.6,
        linestyle="-.",
        label="Generalized ProbStar",
        zorder=4
    )


    # ========================================================
    # Axis formatting
    # ========================================================

    ax.set_xlim(
        -1.0,
        1.0
    )

    ax.set_xticks([
        -1.0,
        -0.5,
         0.0,
         0.5,
         1.0
    ])

    ax.set_xlabel(
        rf"$\alpha_{{{dim + 1}}}$"
    )


    # ========================================================
    # Light grid
    # ========================================================

    ax.grid(
        axis="both",
        which="major",
        linewidth=0.4,
        linestyle="--",
        alpha=0.25,
        zorder=0
    )


    # ========================================================
    # Clean publication appearance
    # ========================================================

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.tick_params(
        axis="both",
        which="major",
        width=0.8,
        length=3
    )


# ============================================================
# 6. Remove unused sixth panel
# ============================================================

axes[5].axis("off")


# ============================================================
# 7. Y-axis labels
#
# One label for each row.
# ============================================================

axes[0].set_ylabel(
    "Probability Density"
)

axes[3].set_ylabel(
    "Probability Density"
)


# ============================================================
# 8. Shared legend
# ============================================================

handles, labels = (
    axes[0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.00),
    ncol=3,
    frameon=False,
    handlelength=2.8,
    columnspacing=1.5,
    handletextpad=0.6
)


# ============================================================
# 9. Layout
# ============================================================

fig.tight_layout(
    rect=[0, 0, 1, 0.91],
    w_pad=1.0,
    h_pad=0.8
)


# ============================================================
# 10. Save publication-quality figures
#
# PDF: use in the TACAS paper
# PNG: high-resolution preview / backup
# ============================================================

fig.savefig(
    "distribution_marginals_3x2.pdf",
    format="pdf",
    bbox_inches="tight"
)

fig.savefig(
    "distribution_marginals_3x2.png",
    format="png",
    dpi=600,
    bbox_inches="tight"
)


# ============================================================
# 11. Display
# ============================================================

plt.show()