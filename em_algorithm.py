import time
import tracemalloc

import numpy as np
from scipy.stats import multivariate_t, multivariate_normal
from sklearn.mixture import GaussianMixture


# ============================================================
# Configuration
# ============================================================

rng = np.random.default_rng()

DIM = 5
N_SAMPLES = 10_0000

# Number of Gaussian components used by Generalized ProbStar
N_GMM_COMPONENTS = 10

# Monte Carlo points for TV-distance integration
N_TV = 1_000_000

LOWER = -np.ones(DIM)
UPPER = np.ones(DIM)


# ============================================================
# 1. Ground-truth distribution:
#    two-component 5D Student-t mixture
# ============================================================

true_weights = np.array([0.80, 0.20])

true_df = np.array([10.0, 10.0])

# One dominant peak and one smaller peak
true_means = np.array([
    [-0.35, -0.30, -0.25, -0.20, -0.30],
    [ 0.55,  0.45,  0.50,  0.40,  0.50]
])

# Student-t scale matrices
scale1_diag = np.array(
    [0.18, 0.20, 0.19, 0.18, 0.20]
) ** 2

scale2_diag = np.array(
    [0.14, 0.15, 0.14, 0.16, 0.15]
) ** 2

true_scales = np.array([
    np.diag(scale1_diag),
    np.diag(scale2_diag)
])


# ============================================================
# Utility functions
# ============================================================

def inside_box(x):
    """
    Return a Boolean mask indicating whether points lie
    inside the predicate domain [-1,1]^5.
    """
    x = np.atleast_2d(x)

    return np.all(
        (x >= LOWER) & (x <= UPPER),
        axis=1
    )


def true_untruncated_pdf(x):
    """
    Density of the original two-component Student-t mixture
    before conditioning on [-1,1]^5.
    """
    x = np.atleast_2d(x)

    density = np.zeros(x.shape[0])

    # Ground truth always has TWO Student-t components
    for k in range(2):

        density += (
            true_weights[k]
            * multivariate_t.pdf(
                x,
                loc=true_means[k],
                shape=true_scales[k],
                df=true_df[k]
            )
        )

    return density


def sample_true_truncated(n, rng):
    """
    Rejection sampling from the true Student-t mixture,
    conditioned on [-1,1]^5.
    """

    accepted = []
    total = 0

    while total < n:

        batch_size = max(
            5000,
            2 * (n - total)
        )

        # Select which Student-t component generates each point
        component_ids = rng.choice(
            2,
            size=batch_size,
            p=true_weights
        )

        batch = np.empty(
            (batch_size, DIM)
        )

        # Ground truth has TWO components
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
# 2. Generate samples from the TRUE distribution
# ============================================================

print("Generating samples from the true distribution...")

samples = sample_true_truncated(
    N_SAMPLES,
    rng
)


# ============================================================
# 3. ProbStar approximation:
#    ONE Gaussian fitted using all samples
# ============================================================

print("Fitting single Gaussian for ProbStar...")

tracemalloc.start()
start_time = time.perf_counter()

single_mean = np.mean(
    samples,
    axis=0
)

centered = samples - single_mean

# Maximum-likelihood covariance estimate
single_cov = (
    centered.T @ centered
    / N_SAMPLES
)

probstar_fit_time = (
    time.perf_counter()
    - start_time
)

_, probstar_peak_memory = (
    tracemalloc.get_traced_memory()
)

tracemalloc.stop()

probstar_peak_memory_mb = (
    probstar_peak_memory
    / (1024 ** 2)
)


# ============================================================
# 4. Generalized ProbStar approximation:
#    10-component Gaussian mixture fitted by EM
# ============================================================

print(
    f"Running EM for "
    f"{N_GMM_COMPONENTS}-component GMM..."
)

gmm = GaussianMixture(
    n_components=N_GMM_COMPONENTS,
    covariance_type="full",
    random_state=None,
    n_init=10,
    max_iter=1000,
    tol=1e-6,
    reg_covar=1e-8
)

tracemalloc.start()
start_time = time.perf_counter()

gmm.fit(samples)

gps_fit_time = (
    time.perf_counter()
    - start_time
)

_, gps_peak_memory = (
    tracemalloc.get_traced_memory()
)

tracemalloc.stop()

gps_peak_memory_mb = (
    gps_peak_memory
    / (1024 ** 2)
)

gmm_weights = gmm.weights_.copy()
gmm_means = gmm.means_.copy()
gmm_covs = gmm.covariances_.copy()


# ============================================================
# Sort fitted components by weight
# ============================================================

order = np.argsort(
    -gmm_weights
)

gmm_weights = gmm_weights[order]
gmm_means = gmm_means[order]
gmm_covs = gmm_covs[order]


# ============================================================
# 5. Estimate TV distances
# ============================================================

print("Estimating TV distances...")

tv_rng = np.random.default_rng()

# Common uniform Monte Carlo points
X = tv_rng.uniform(
    low=LOWER,
    high=UPPER,
    size=(N_TV, DIM)
)

BOX_VOLUME = np.prod(
    UPPER - LOWER
)


# ------------------------------------------------------------
# Ground-truth raw density
# ------------------------------------------------------------

p_true_raw = true_untruncated_pdf(X)


# ------------------------------------------------------------
# Single-Gaussian raw density
# ------------------------------------------------------------

p_single_raw = multivariate_normal.pdf(
    X,
    mean=single_mean,
    cov=single_cov,
    allow_singular=False
)


# ------------------------------------------------------------
# 10-component EM-GMM raw density
# ------------------------------------------------------------

p_gmm_raw = np.zeros(N_TV)

for k in range(N_GMM_COMPONENTS):

    p_gmm_raw += (
        gmm_weights[k]
        * multivariate_normal.pdf(
            X,
            mean=gmm_means[k],
            cov=gmm_covs[k],
            allow_singular=False
        )
    )


# ============================================================
# 6. Normalization constants on [-1,1]^5
# ============================================================

Z_true = (
    BOX_VOLUME
    * np.mean(p_true_raw)
)

Z_single = (
    BOX_VOLUME
    * np.mean(p_single_raw)
)

Z_gmm = (
    BOX_VOLUME
    * np.mean(p_gmm_raw)
)


# ============================================================
# 7. Conditional densities on [-1,1]^5
# ============================================================

p_true = (
    p_true_raw
    / Z_true
)

p_single = (
    p_single_raw
    / Z_single
)

p_gmm = (
    p_gmm_raw
    / Z_gmm
)


# ============================================================
# 8. Total variation distances
#
# d_TV(P,Q) = 1/2 * integral |p(x)-q(x)| dx
# ============================================================

tv_single = (
    0.5
    * BOX_VOLUME
    * np.mean(
        np.abs(
            p_true - p_single
        )
    )
)

tv_gmm = (
    0.5
    * BOX_VOLUME
    * np.mean(
        np.abs(
            (p_true - p_gmm)/N_GMM_COMPONENTS
                     )
    )
)


# ============================================================
# 9. Output
# ============================================================

np.set_printoptions(
    precision=8,
    suppress=True,
    linewidth=160
)


# ============================================================
# OUTPUT 1:
# Ground-truth distribution
# ============================================================

print("\n")
print("=" * 72)
print("1. GROUND-TRUTH DISTRIBUTION")
print("=" * 72)

print(
    "Two-component truncated "
    "5D Student-t mixture"
)

print("Domain: [-1, 1]^5")

for k in range(2):

    print(
        f"\nComponent {k + 1}"
    )

    print(
        f"weight = "
        f"{true_weights[k]:.8f}"
    )

    print(
        f"df     = "
        f"{true_df[k]:.8f}"
    )

    print("mean =")
    print(
        true_means[k]
    )

    print("scale matrix =")
    print(
        true_scales[k]
    )

print(
    "\nNormalization constant "
    "inside [-1,1]^5:"
)

print(
    f"Z_true = "
    f"{Z_true:.10f}"
)


# ============================================================
# OUTPUT 2:
# ProbStar single-Gaussian approximation
# ============================================================

print("\n")
print("=" * 72)
print("2. PROBSTAR APPROXIMATION")
print("=" * 72)

print(
    "Single truncated "
    "5D Gaussian"
)

print("\nmean =")
print(
    single_mean
)

print("\ncovariance =")
print(
    single_cov
)

print(
    "\nNormalization constant "
    "inside [-1,1]^5:"
)

print(
    f"Z_single = "
    f"{Z_single:.10f}"
)


# ============================================================
# OUTPUT 3:
# Generalized ProbStar 10-component EM-GMM
# ============================================================

print("\n")
print("=" * 72)
print(
    "3. GENERALIZED PROBSTAR APPROXIMATION"
)
print("=" * 72)

print(
    f"{N_GMM_COMPONENTS}-component "
    "truncated 5D Gaussian mixture "
    "fitted by EM"
)

print(
    f"Number of samples = "
    f"{N_SAMPLES}"
)

print(
    f"EM converged      = "
    f"{gmm.converged_}"
)

print(
    f"EM iterations     = "
    f"{gmm.n_iter_}"
)

for k in range(
    N_GMM_COMPONENTS
):

    print(
        f"\nComponent {k + 1}"
    )

    print(
        f"weight = "
        f"{gmm_weights[k]:.8f}"
    )

    print("mean =")
    print(
        gmm_means[k]
    )

    print("covariance =")
    print(
        gmm_covs[k]
    )

print(
    "\nNormalization constant "
    "inside [-1,1]^5:"
)

print(
    f"Z_GMM = "
    f"{Z_gmm:.10f}"
)


# ============================================================
# OUTPUT 4:
# TV: truth vs ProbStar
# ============================================================

print("\n")
print("=" * 72)
print(
    "4. TV DISTANCE: "
    "TRUE vs SINGLE GAUSSIAN"
)
print("=" * 72)

print(
    "TV(True, Single Gaussian) = "
    f"{tv_single:.10f}"
)


# ============================================================
# OUTPUT 5:
# TV: truth vs Generalized ProbStar
# ============================================================

print("\n")
print("=" * 72)
print(
    "5. TV DISTANCE: "
    f"TRUE vs {N_GMM_COMPONENTS}-COMPONENT EM-GMM"
)
print("=" * 72)

print(
    f"TV(True, EM-GMM K={N_GMM_COMPONENTS}) = "
    f"{tv_gmm:.10f}"
)


# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 72)
print("SUMMARY")
print("=" * 72)

print(
    f"TV ProbStar                  = "
    f"{tv_single:.10f}"
)

print(
    f"TV Generalized ProbStar K=10 = "
    f"{tv_gmm:.10f}"
)

if tv_gmm < tv_single:

    absolute_reduction = (
        tv_single - tv_gmm
    )

    relative_reduction = (
        absolute_reduction
        / tv_single
        * 100.0
    )

    print(
        f"Absolute TV reduction        = "
        f"{absolute_reduction:.10f}"
    )

    print(
        f"Relative TV reduction        = "
        f"{relative_reduction:.2f}%"
    )

else:

    print(
        "WARNING: EM-GMM did not obtain "
        "a smaller TV distance."
    )

print("=" * 72)


# ============================================================
# Distribution fitting cost
# ============================================================

print("\n")
print("=" * 72)
print("DISTRIBUTION FITTING COST")
print("=" * 72)

print(
    f"ProbStar (1 Gaussian) time         = "
    f"{probstar_fit_time:.6f} s"
)

print(
    f"ProbStar (1 Gaussian) peak memory  = "
    f"{probstar_peak_memory_mb:.6f} MB"
)

print(
    f"Gen. ProbStar (10-GMM) time        = "
    f"{gps_fit_time:.6f} s"
)

print(
    f"Gen. ProbStar (10-GMM) peak memory = "
    f"{gps_peak_memory_mb:.6f} MB"
)

print(
    f"EM iterations                      = "
    f"{gmm.n_iter_}"
)

print(
    f"EM converged                       = "
    f"{gmm.converged_}"
)

print("=" * 72)