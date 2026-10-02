#!/usr/bin/env python3

"""
ACAS Xu Property 3
Direct probability computation on SAVED exact reachable Stars.

NO network propagation.
NO ReLU propagation.
NO distribution fitting.
NO Monte Carlo.
NO QMC.

Three fixed predicate distributions:

1. True truncated 2-component Student-t mixture
2. Original ProbStar truncated single Gaussian
3. Generalized ProbStar truncated K=10 Gaussian mixture

Probability is evaluated directly on the saved Star geometry.
"""

import csv
import pickle
import time
import tracemalloc
from pathlib import Path

import numpy as np

from scipy.optimize import linprog
from scipy.stats import multivariate_normal, multivariate_t
from numpy.polynomial.legendre import leggauss


# ============================================================
# Configuration
# ============================================================

RESULTS_ROOT = Path("results")
START_NETWORK = 1
END_NETWORK = 45

PKL_NAME = "acasxu_reachable_stars.pkl"
NETWORK_RESULT_CSV = "acasxu_probability_results.csv"
MASTER_RESULT_CSV = RESULTS_ROOT / "acasxu_all_probability_results.csv"

DIM = 5

LB = -np.ones(DIM)
UB =  np.ones(DIM)

# Deterministic Gauss-Legendre order.
#
# 12^5 = 248,832 quadrature points.
#
# Increase to 15 or 17 later for convergence checking.
GL_ORDER = 12

TOL = 1e-10


# ============================================================
# ACAS Xu Property 3
#
# Unsafe:
#
# y0 <= y1
# y0 <= y2
# y0 <= y3
# y0 <= y4
#
# H y <= g
# ============================================================

# Standard ACAS Xu Property-3 unsafe output region:
# y0 <= y1, y0 <= y2, y0 <= y3, y0 <= y4.
# Equivalently, H y <= g.
H = np.array([
    [1.0, -1.0,  0.0,  0.0,  0.0],
    [1.0,  0.0, -1.0,  0.0,  0.0],
    [1.0,  0.0,  0.0, -1.0,  0.0],
    [1.0,  0.0,  0.0,  0.0, -1.0]
])

g = np.zeros(4)


# ============================================================
# 1. TRUE DISTRIBUTION
#
# Two-component truncated 5D Student-t mixture
# ============================================================

TRUE_WEIGHTS = np.array([
    0.8,
    0.2
])

TRUE_DF = 10.0

TRUE_MEANS = np.array([
    [-0.35, -0.30, -0.25, -0.20, -0.30],
    [ 0.55,  0.45,  0.50,  0.40,  0.50]
])

TRUE_COVS = np.array([

    np.diag([
        0.0324,
        0.0400,
        0.0361,
        0.0324,
        0.0400
    ]),

    np.diag([
        0.0196,
        0.0225,
        0.0196,
        0.0256,
        0.0225
    ])
])

Z_TRUE = 0.9777465865


# ============================================================
# 2. ORIGINAL PROBSTAR
#
# Single truncated 5D Gaussian
# ============================================================

PS_MEAN = np.array([
    -0.17382733,
    -0.15508302,
    -0.10549723,
    -0.08245848,
    -0.14647688
])

PS_COV = np.array([
    [0.16098538, 0.10517317, 0.10604200, 0.08393378, 0.11061816],
    [0.10517317, 0.13359542, 0.08901846, 0.07019068, 0.09347490],
    [0.10604200, 0.08901846, 0.12762315, 0.07053787, 0.09450510],
    [0.08393378, 0.07019068, 0.07053787, 0.09424588, 0.07321418],
    [0.11061816, 0.09347490, 0.09450510, 0.07321418, 0.14211245]
])

Z_SINGLE = 0.9635233710


# ============================================================
# 3. GENERALIZED PROBSTAR
#
# Fixed K=10 truncated Gaussian mixture
# ============================================================

GPS_WEIGHTS = np.array([
    0.28300800,
    0.24510673,
    0.13520286,
    0.12050350,
    0.07273553,
    0.05958471,
    0.04617820,
    0.01680844,
    0.01403072,
    0.00684131
])

GPS_WEIGHTS = (
    GPS_WEIGHTS
    / GPS_WEIGHTS.sum()
)


GPS_MEANS = np.array([

    [-0.34558652, -0.30412842, -0.24923874, -0.18281973, -0.31633116],

    [-0.38146254, -0.27427800, -0.29571725, -0.19167988, -0.38197128],

    [ 0.55150431,  0.45262660,  0.50743149,  0.39660933,  0.50053115],

    [-0.37757650, -0.30791713, -0.24383230, -0.22040672, -0.19467421],

    [-0.27915168, -0.37077062, -0.19831213, -0.25004395, -0.14851801],

    [ 0.52965069,  0.45055315,  0.48575273,  0.39383445,  0.46604986],

    [-0.25525745, -0.25305826, -0.19763299, -0.12080802, -0.21604355],

    [-0.37078501, -0.30628064, -0.16261368, -0.20351010, -0.52456275],

    [-0.27908779, -0.35899413, -0.14668842, -0.39081333, -0.18644871],

    [-0.13295655, -0.56761481, -0.33667030, -0.23337530, -0.45320787]
])


GPS_COVS = np.array([

    [
        [ 0.01979855,  0.00006393,  0.00051066,  0.00281928, -0.00221216],
        [ 0.00006393,  0.02955551,  0.00018640,  0.00017041,  0.00350867],
        [ 0.00051066,  0.00018640,  0.02563112, -0.00017909, -0.00006009],
        [ 0.00281928,  0.00017041, -0.00017909,  0.02607916,  0.00314249],
        [-0.00221216,  0.00350867, -0.00006009,  0.00314249,  0.02251244]
    ],

    [
        [ 0.04293289, -0.00021238, -0.00343148, -0.00258473, -0.00266919],
        [-0.00021238,  0.05128917,  0.00186295,  0.00038233, -0.00063810],
        [-0.00343148,  0.00186295,  0.03602537,  0.00602461, -0.00067230],
        [-0.00258473,  0.00038233,  0.00602461,  0.03924504, -0.00208983],
        [-0.00266919, -0.00063810, -0.00067230, -0.00208983,  0.04701325]
    ],

    [
        [ 0.01627100, -0.00069143, -0.00140774,  0.00128459,  0.00063360],
        [-0.00069143,  0.01873681,  0.00013508,  0.00072080,  0.00024266],
        [-0.00140774,  0.00013508,  0.01655289,  0.00061736,  0.00003026],
        [ 0.00128459,  0.00072080,  0.00061736,  0.02110142, -0.00142189],
        [ 0.00063360,  0.00024266,  0.00003026, -0.00142189,  0.01854472]
    ],

    [
        [ 0.04149231,  0.00408128,  0.00486950,  0.00299395,  0.01552095],
        [ 0.00408128,  0.06293590,  0.00291951, -0.00766230,  0.00062837],
        [ 0.00486950,  0.00291951,  0.06160253, -0.00969989, -0.00795539],
        [ 0.00299395, -0.00766230, -0.00969989,  0.05647876,  0.00201741],
        [ 0.01552095,  0.00062837, -0.00795539,  0.00201741,  0.05774970]
    ],

    [
        [ 0.04141458,  0.00206272, -0.00474192, -0.00148020, -0.01179525],
        [ 0.00206272,  0.03177307,  0.00060621,  0.00590184,  0.00156027],
        [-0.00474192,  0.00060621,  0.02453604, -0.00102905, -0.00423807],
        [-0.00148020,  0.00590184, -0.00102905,  0.01563799,  0.00529446],
        [-0.01179525,  0.00156027, -0.00423807,  0.00529446,  0.02212083]
    ],

    [
        [ 0.03717413,  0.00118489,  0.00148188,  0.00086129, -0.00360670],
        [ 0.00118489,  0.04248193, -0.00308997, -0.00320486, -0.00271715],
        [ 0.00148188, -0.00308997,  0.03842504, -0.00042302,  0.00376254],
        [ 0.00086129, -0.00320486, -0.00042302,  0.05174106,  0.00240626],
        [-0.00360670, -0.00271715,  0.00376254,  0.00240626,  0.04364519]
    ],

    [
        [ 0.09571889,  0.00974677, -0.00713545, -0.00477402, -0.02887704],
        [ 0.00974677,  0.11369134,  0.00645309,  0.00699384,  0.00097629],
        [-0.00713545,  0.00645309,  0.11896911,  0.00231065,  0.01613714],
        [-0.00477402,  0.00699384,  0.00231065,  0.10154114, -0.01797944],
        [-0.02887704,  0.00097629,  0.01613714, -0.01797944,  0.09686887]
    ],

    [
        [ 0.04616753, -0.02088663,  0.03603102,  0.00185709,  0.01949255],
        [-0.02088663,  0.09382258, -0.05226160, -0.04154559,  0.00634369],
        [ 0.03603102, -0.05226160,  0.08609198,  0.00894669, -0.01575531],
        [ 0.00185709, -0.04154559,  0.00894669,  0.07446758,  0.00915059],
        [ 0.01949255,  0.00634369, -0.01575531,  0.00915059,  0.03745857]
    ],

    [
        [ 0.04010451, -0.01878237, -0.00328250,  0.01420448, -0.00606186],
        [-0.01878237,  0.01956414,  0.00408125,  0.00346255,  0.00823425],
        [-0.00328250,  0.00408125,  0.01207111,  0.00449625,  0.02315015],
        [ 0.01420448,  0.00346255,  0.00449625,  0.03723127,  0.00724532],
        [-0.00606186,  0.00823425,  0.02315015,  0.00724532,  0.04830860]
    ],

    [
        [ 0.02981105, -0.00005920,  0.02488262,  0.03946179,  0.00791451],
        [-0.00005920,  0.03922133,  0.02172393,  0.00435795, -0.02542279],
        [ 0.02488262,  0.02172393,  0.08562576,  0.02852525, -0.02189836],
        [ 0.03946179,  0.00435795,  0.02852525,  0.06464091,  0.02990876],
        [ 0.00791451, -0.02542279, -0.02189836,  0.02990876,  0.06970784]
    ]
])

Z_GMM = 0.9813868733


# ============================================================
# Load saved exact Stars
# ============================================================

def load_stars(pkl_file):

    print(f"\nLoading {pkl_file} ...")

    with open(pkl_file, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        stars = obj

    elif isinstance(obj, dict):

        for key in [
            "stars",
            "reachable_stars",
            "final_stars"
        ]:
            if key in obj:
                stars = obj[key]
                break
        else:
            raise RuntimeError(
                "Cannot find Stars in pickle."
            )

    else:
        raise RuntimeError(
            f"Unknown pickle format: {type(obj)}"
        )

    print(
        f"Loaded final Stars: {len(stars):,}"
    )

    return stars


# ============================================================
# Star field helper
# ============================================================

def get_array(star, name):

    x = np.asarray(
        star[name],
        dtype=float
    )

    return x


# ============================================================
# Property-3 unsafe intersection
# ============================================================

def make_unsafe_polytope(star):

    c = get_array(
        star,
        "c"
    ).reshape(-1)

    V = get_array(
        star,
        "V"
    )

    C = get_array(
        star,
        "C"
    )

    d = get_array(
        star,
        "d"
    ).reshape(-1)

    lb = get_array(
        star,
        "lb"
    ).reshape(-1)

    ub = get_array(
        star,
        "ub"
    ).reshape(-1)


    if C.size == 0:

        C = np.zeros(
            (0, DIM)
        )

    elif C.ndim == 1:

        C = C.reshape(
            1,
            -1
        )


    # H(c + V alpha) <= g
    #
    # HV alpha <= g - Hc

    C_unsafe = H @ V

    d_unsafe = (
        g
        - H @ c
    )


    C_final = np.vstack([
        C,
        C_unsafe
    ])

    d_final = np.concatenate([
        d,
        d_unsafe
    ])


    return (
        C_final,
        d_final,
        lb,
        ub
    )


# ============================================================
# LP feasibility
# ============================================================

def feasible_polytope(
    C,
    d,
    lb,
    ub
):

    result = linprog(

        c=np.zeros(DIM),

        A_ub=C,

        b_ub=d,

        bounds=list(
            zip(lb, ub)
        ),

        method="highs"
    )

    return result.success


# ============================================================
# Prepare unsafe branch geometry ONCE
# ============================================================

def prepare_unsafe_polytopes(
    stars
):

    print()
    print("=" * 80)
    print(
        "Preparing Property-3 unsafe intersections"
    )
    print("=" * 80)

    start = time.perf_counter()

    unsafe = []


    for i, star in enumerate(stars):

        (
            C,
            d,
            lb,
            ub
        ) = make_unsafe_polytope(
            star
        )


        if feasible_polytope(
            C,
            d,
            lb,
            ub
        ):

            unsafe.append(
                (
                    C,
                    d,
                    lb,
                    ub
                )
            )


        if (i + 1) % 5000 == 0:

            print(
                f"{i+1:,} / "
                f"{len(stars):,}"
            )


    elapsed = (
        time.perf_counter()
        - start
    )


    print()
    print(
        f"Total final Stars       : "
        f"{len(stars):,}"
    )

    print(
        f"Unsafe-intersecting Stars: "
        f"{len(unsafe):,}"
    )

    print(
        f"Geometry filtering time : "
        f"{elapsed:.3f} s"
    )


    return unsafe, elapsed


# ============================================================
# Deterministic Gauss-Legendre grid
#
# Domain = [-1,1]^5
# ============================================================

def build_quadrature():

    print()
    print("=" * 80)
    print(
        "Building deterministic Gauss-Legendre quadrature"
    )
    print("=" * 80)


    nodes_1d, weights_1d = (
        leggauss(GL_ORDER)
    )


    grids = np.meshgrid(
        nodes_1d,
        nodes_1d,
        nodes_1d,
        nodes_1d,
        nodes_1d,
        indexing="ij"
    )


    weight_grids = np.meshgrid(
        weights_1d,
        weights_1d,
        weights_1d,
        weights_1d,
        weights_1d,
        indexing="ij"
    )


    points = np.stack(
        [
            x.reshape(-1)
            for x in grids
        ],
        axis=1
    )


    weights = np.ones(
        len(points)
    )


    for w in weight_grids:

        weights *= (
            w.reshape(-1)
        )


    print(
        f"Quadrature order  : {GL_ORDER}"
    )

    print(
        f"Quadrature points : "
        f"{len(points):,}"
    )


    return points, weights


# ============================================================
# Fixed densities
# ============================================================

def true_density(points):

    density = np.zeros(
        len(points)
    )


    for weight, mean, cov in zip(
        TRUE_WEIGHTS,
        TRUE_MEANS,
        TRUE_COVS
    ):

        density += (
            weight
            *
            multivariate_t.pdf(
                points,
                loc=mean,
                shape=cov,
                df=TRUE_DF
            )
        )


    # Truncated / conditioned
    return (
        density
        / Z_TRUE
    )


def probstar_density(points):

    density = (
        multivariate_normal.pdf(
            points,
            mean=PS_MEAN,
            cov=PS_COV
        )
    )


    return (
        density
        / Z_SINGLE
    )


def gps_density(points):

    density = np.zeros(
        len(points)
    )


    for weight, mean, cov in zip(
        GPS_WEIGHTS,
        GPS_MEANS,
        GPS_COVS
    ):

        density += (
            weight
            *
            multivariate_normal.pdf(
                points,
                mean=mean,
                cov=cov
            )
        )


    return (
        density
        / Z_GMM
    )


# ============================================================
# Direct deterministic probability
# ============================================================

def compute_probability(
    unsafe_polytopes,
    points,
    quad_weights,
    density,
    name
):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)


    # Statistics only: preserve the original probability computation.
    tracemalloc.start()
    start = time.perf_counter()

    total_probability = 0.0


    for i, (
        C,
        d,
        lb,
        ub
    ) in enumerate(
        unsafe_polytopes
    ):


        # Quadrature points are already in [-1,1]^5.
        #
        # Check the Star constraints directly.

        inside = np.all(

            points @ C.T
            <=
            d[None, :]
            + TOL,

            axis=1
        )


        if np.any(inside):

            p = np.sum(
                quad_weights[inside]
                *
                density[inside]
            )

            total_probability += p



    elapsed = (
        time.perf_counter()
        - start
    )

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory / (1024 ** 2)


    print()
    print(
        f"P(unsafe) = "
        f"{total_probability:.12e}"
    )

    print(
        f"Probability time = "
        f"{elapsed:.3f} s"
    )

    print(
        f"Peak memory = "
        f"{peak_memory_mb:.3f} MB"
    )


    return {
        "probability":
            float(total_probability),

        "time":
            elapsed,

        "peak_memory_mb":
            peak_memory_mb
    }


# ============================================================
# Main
# ============================================================
# ============================================================
# CSV helpers
# ============================================================

RESULT_FIELDS = [
    "network_id",
    "num_final_stars",
    "num_unsafe_stars",
    "geometry_time_s",
    "p_true",
    "p_probstar",
    "p_gps",
    "probstar_abs_error",
    "gps_abs_error",
    "gps_error_reduction_percent",
    "true_probability_time_s",
    "true_peak_memory_mb",
    "probstar_probability_time_s",
    "probstar_peak_memory_mb",
    "gps_probability_time_s",
    "gps_peak_memory_mb",
    "gps_to_probstar_time_ratio"
]


def write_result_csv(path, rows):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# One network
# ============================================================

def run_one_network(
    network_id,
    points,
    quad_weights,
    density_true,
    density_ps,
    density_gps
):

    network_dir = RESULTS_ROOT / str(network_id)
    pkl_file = network_dir / PKL_NAME

    if not pkl_file.exists():
        raise FileNotFoundError(
            f"Network {network_id}: cannot find {pkl_file}"
        )

    print("\n" + "#" * 125)
    print(f"NETWORK {network_id} / {END_NETWORK}")
    print(f"Input : {pkl_file}")
    print("#" * 125)

    stars = load_stars(pkl_file)
    num_final_stars = len(stars)

    unsafe_polytopes, geometry_time = prepare_unsafe_polytopes(stars)
    num_unsafe_stars = len(unsafe_polytopes)

    # If the exact reachable set has no intersection with the
    # Property-3 unsafe region, the probability is exactly zero
    # under all three predicate distributions.
    if num_unsafe_stars == 0:

        result_true = {
            "probability": 0.0,
            "time": 0.0,
            "peak_memory_mb": 0.0
        }
        result_ps = {
            "probability": 0.0,
            "time": 0.0,
            "peak_memory_mb": 0.0
        }
        result_gps = {
            "probability": 0.0,
            "time": 0.0,
            "peak_memory_mb": 0.0
        }

    else:

        result_true = compute_probability(
            unsafe_polytopes,
            points,
            quad_weights,
            density_true,
            "TRUE STUDENT-t MIXTURE"
        )

        result_ps = compute_probability(
            unsafe_polytopes,
            points,
            quad_weights,
            density_ps,
            "ORIGINAL PROBSTAR"
        )

        result_gps = compute_probability(
            unsafe_polytopes,
            points,
            quad_weights,
            density_gps,
            "GENERALIZED PROBSTAR K=10"
        )

    p_true = result_true["probability"]
    p_ps = result_ps["probability"]
    p_gps = result_gps["probability"]

    error_ps = abs(p_ps - p_true)
    error_gps = abs(p_gps - p_true)

    if error_ps > 0:
        reduction = (1.0 - error_gps / error_ps) * 100.0
    else:
        reduction = 0.0

    if result_ps["time"] > 0:
        time_ratio = result_gps["time"] / result_ps["time"]
    else:
        time_ratio = 0.0

    row = {
        "network_id": network_id,
        "num_final_stars": num_final_stars,
        "num_unsafe_stars": num_unsafe_stars,
        "geometry_time_s": geometry_time,
        "p_true": p_true,
        "p_probstar": p_ps,
        "p_gps": p_gps,
        "probstar_abs_error": error_ps,
        "gps_abs_error": error_gps,
        "gps_error_reduction_percent": reduction,
        "true_probability_time_s": result_true["time"],
        "true_peak_memory_mb": result_true["peak_memory_mb"],
        "probstar_probability_time_s": result_ps["time"],
        "probstar_peak_memory_mb": result_ps["peak_memory_mb"],
        "gps_probability_time_s": result_gps["time"],
        "gps_peak_memory_mb": result_gps["peak_memory_mb"],
        "gps_to_probstar_time_ratio": time_ratio
    }

    # Save this network's result immediately in its own folder.
    network_csv = network_dir / NETWORK_RESULT_CSV
    write_result_csv(network_csv, [row])

    print("\n" + "=" * 125)
    print(f"NETWORK {network_id} FINAL RESULT")
    print("=" * 125)
    print(f"Final Stars             : {num_final_stars:,}")
    print(f"Unsafe-intersecting     : {num_unsafe_stars:,}")
    print(f"P_true(unsafe)          : {p_true:.12e}")
    print(f"P_ProbStar(unsafe)      : {p_ps:.12e}")
    print(f"P_GPS(unsafe)           : {p_gps:.12e}")
    print(f"ProbStar abs. error     : {error_ps:.12e}")
    print(f"GPS abs. error          : {error_gps:.12e}")
    print(f"GPS error reduction     : {reduction:.3f}%")
    print(f"True time / memory      : {result_true['time']:.3f} s / {result_true['peak_memory_mb']:.3f} MB")
    print(f"ProbStar time / memory  : {result_ps['time']:.3f} s / {result_ps['peak_memory_mb']:.3f} MB")
    print(f"GPS time / memory       : {result_gps['time']:.3f} s / {result_gps['peak_memory_mb']:.3f} MB")
    print(f"Saved                   : {network_csv}")
    print("=" * 125)

    return row


# ============================================================
# Main batch
# ============================================================

def main():

    if not (1 <= START_NETWORK <= END_NETWORK <= 45):
        raise ValueError(
            "Require 1 <= START_NETWORK <= END_NETWORK <= 45."
        )

    print("=" * 100)
    print("ACAS Xu Property 3 -- BATCH DIRECT PROBABILITY COMPUTATION")
    print("=" * 100)
    print("Unsafe region: y0 <= y1, y0 <= y2, y0 <= y3, y0 <= y4")
    print(f"Networks     : {START_NETWORK} ... {END_NETWORK}")
    print(f"Results root : {RESULTS_ROOT}")
    print("No network propagation / no ReLU propagation / no fitting / no MC / no QMC.")

    # The quadrature grid and the three fixed predicate densities
    # are identical for all 45 networks, so construct them only once.
    points, quad_weights = build_quadrature()

    print("\nEvaluating the three fixed densities on the quadrature grid once...")
    density_true = true_density(points)
    density_ps = probstar_density(points)
    density_gps = gps_density(points)

    all_rows = []

    for network_id in range(START_NETWORK, END_NETWORK + 1):

        row = run_one_network(
            network_id,
            points,
            quad_weights,
            density_true,
            density_ps,
            density_gps
        )

        all_rows.append(row)

        # Update the master CSV after every completed network so that
        # already-finished results survive an interruption.
        write_result_csv(MASTER_RESULT_CSV, all_rows)
        print(f"Master CSV updated: {MASTER_RESULT_CSV}")

    print("\n" + "=" * 100)
    print("ALL REQUESTED NETWORKS COMPLETED")
    print(f"Master CSV: {MASTER_RESULT_CSV}")
    print("=" * 100)


if __name__ == "__main__":
    main()
