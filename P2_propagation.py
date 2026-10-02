#!/usr/bin/env python3

"""
Stage 1 -- Full ACAS Xu Exact-Star Reachability
================================================

Purpose
-------
1. Load ACAS Xu network.
2. Construct the Property-2 input Star.
3. Propagate the Star through the COMPLETE network:
       6 hidden affine + ReLU layers
       1 final affine output layer
4. No probability distribution is used.
5. Save final Star geometry to:
       - PKL : primary file for Stage 2
       - CSV : human-readable / independent reload
6. Record:
       - branch counts
       - affine/ReLU runtime
       - total runtime
       - Python peak memory
       - process RSS
7. Reload and verify saved geometry.

Star representation
-------------------
    x = c + V alpha
    C alpha <= d
    lb <= alpha <= ub

The probability distribution of alpha is deliberately excluded.
"""

from pathlib import Path
import csv
import json
import os
import pickle
import resource
import time
import tracemalloc

import numpy as np
import scipy.io as sio
from scipy.optimize import linprog


# ============================================================
# Configuration
# ============================================================

# Batch configuration: network IDs use row-major order:
# 1=(1,1), 2=(1,2), ..., 9=(1,9), 10=(2,1), ..., 45=(5,9).
TARGET_NETWORKS = [11, 19, 24, 25, 28, 34]
RESULTS_ROOT = Path("results_property2")

# These names are reassigned for each network inside run_one_network().
MAT_FILE = "ACASXU_run2a_1_1_batch_2000.mat"

PKL_FILE = "acasxu_reachable_stars.pkl"
CSV_FILE = "acasxu_reachable_stars.csv"

LAYER_CSV = "acasxu_stage1_layers.csv"
SUMMARY_TXT = "acasxu_stage1_summary.txt"

# CSV can become large for ~70k Stars.
SAVE_CSV = True

# Verify PKL completely.
VERIFY_PKL = True

# CSV reload is slower, but useful for this first full run.
VERIFY_CSV = True

# Print progress during serialization.
SAVE_PROGRESS_INTERVAL = 5000


# ============================================================
# ACAS Xu Property 2
# ============================================================

ALPHA_LB = -np.ones(5)
ALPHA_UB = np.ones(5)

PROPERTY2_RAW_LB = np.array([
    55947.691,
    -np.pi,
    -np.pi,
    1145.0,
    0.0
])

PROPERTY2_RAW_UB = np.array([
    60760.0,
    np.pi,
    np.pi,
    1200.0,
    60.0
])

PROPERTY2_MEANS = np.array([
    19791.091,
    0.0,
    0.0,
    650.0,
    600.0
])

PROPERTY2_RANGES = np.array([
    60261.0,
    2.0 * np.pi,
    2.0 * np.pi,
    1100.0,
    1200.0
])

PROPERTY2_X_LB = (
    PROPERTY2_RAW_LB
    - PROPERTY2_MEANS
) / PROPERTY2_RANGES

PROPERTY2_X_UB = (
    PROPERTY2_RAW_UB
    - PROPERTY2_MEANS
) / PROPERTY2_RANGES

PROPERTY2_C = 0.5 * (
    PROPERTY2_X_LB
    + PROPERTY2_X_UB
)

PROPERTY2_V_DIAG = 0.5 * (
    PROPERTY2_X_UB
    - PROPERTY2_X_LB
)

PROPERTY2_V = np.diag(
    PROPERTY2_V_DIAG
)


# ============================================================
# Utility: process memory
# ============================================================

def get_process_rss_mb():
    """
    Current RSS from /proc on Linux.

    This is different from tracemalloc:
      tracemalloc -> Python allocations
      RSS         -> whole process resident memory
    """

    try:
        with open(
            "/proc/self/status",
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if line.startswith("VmRSS:"):

                    kb = float(
                        line.split()[1]
                    )

                    return kb / 1024.0

    except Exception:
        pass

    return float("nan")


def get_max_rss_mb():
    """
    Maximum RSS reported by getrusage.

    On Linux, ru_maxrss is in KB.
    """

    value = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    return float(value) / 1024.0


# ============================================================
# ACAS Xu network loader
# ============================================================

def _mat_cell_to_list(cell):

    arr = np.asarray(cell)

    return [
        np.asarray(
            item,
            dtype=float
        )
        for item in arr.ravel()
    ]


def load_acasxu_mat(mat_path):

    data = sio.loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False
    )

    W = _mat_cell_to_list(
        data["W"]
    )

    b = [
        np.asarray(
            item,
            dtype=float
        ).reshape(-1)
        for item in _mat_cell_to_list(
            data["b"]
        )
    ]

    if "layer_sizes" in data:

        layer_sizes = np.asarray(
            data["layer_sizes"]
        ).astype(int).reshape(-1)

    else:

        layer_sizes = np.array(
            [W[0].shape[1]]
            +
            [
                Wi.shape[0]
                for Wi in W
            ],
            dtype=int
        )

    if len(W) != len(b):

        raise RuntimeError(
            "Number of weight matrices and "
            "bias vectors does not match."
        )

    return {
        "W": W,
        "b": b,
        "layer_sizes": layer_sizes
    }


# ============================================================
# Exact Star
# ============================================================

class ExactStar:

    """
    Star:

        x = c + V alpha

    subject to:

        C alpha <= d
        lb <= alpha <= ub
    """

    def __init__(
        self,
        c,
        V,
        C,
        d,
        lb,
        ub,
        activation_pattern=None
    ):

        self.c = np.asarray(
            c,
            dtype=float
        ).reshape(-1)

        self.V = np.asarray(
            V,
            dtype=float
        )

        self.C = np.asarray(
            C,
            dtype=float
        )

        self.d = np.asarray(
            d,
            dtype=float
        ).reshape(-1)

        self.lb = np.asarray(
            lb,
            dtype=float
        ).reshape(-1)

        self.ub = np.asarray(
            ub,
            dtype=float
        ).reshape(-1)

        if self.V.ndim == 1:
            self.V = self.V.reshape(
                -1,
                1
            )

        if activation_pattern is None:
            self.activation_pattern = []
        else:
            self.activation_pattern = list(
                activation_pattern
            )

        self.dim = len(
            self.c
        )

        self.n_pred = (
            self.V.shape[1]
        )


    def copy(self):

        return ExactStar(
            self.c.copy(),
            self.V.copy(),
            self.C.copy(),
            self.d.copy(),
            self.lb.copy(),
            self.ub.copy(),
            self.activation_pattern.copy()
        )


    def _lp_bounds(self):

        return [
            (
                self.lb[i],
                self.ub[i]
            )
            for i in range(
                self.n_pred
            )
        ]


    def is_empty(self):

        result = linprog(
            c=np.zeros(
                self.n_pred
            ),
            A_ub=self.C,
            b_ub=self.d,
            bounds=self._lp_bounds(),
            method="highs"
        )

        return not result.success


    def affine_map(
        self,
        W,
        b
    ):

        W = np.asarray(
            W,
            dtype=float
        )

        b = np.asarray(
            b,
            dtype=float
        ).reshape(-1)

        return ExactStar(
            W @ self.c + b,
            W @ self.V,
            self.C,
            self.d,
            self.lb,
            self.ub,
            self.activation_pattern
        )


    def get_range_lp(
        self,
        idx
    ):

        v = self.V[
            idx,
            :
        ]

        rmin = linprog(
            c=v,
            A_ub=self.C,
            b_ub=self.d,
            bounds=self._lp_bounds(),
            method="highs"
        )

        rmax = linprog(
            c=-v,
            A_ub=self.C,
            b_ub=self.d,
            bounds=self._lp_bounds(),
            method="highs"
        )

        if (
            not rmin.success
            or
            not rmax.success
        ):

            raise RuntimeError(
                "LP range computation failed."
            )

        lower = (
            self.c[idx]
            + rmin.fun
        )

        upper = (
            self.c[idx]
            - rmax.fun
        )

        return (
            float(lower),
            float(upper)
        )


    @staticmethod
    def step_relu(
        star,
        neuron_index
    ):

        lower, upper = (
            star.get_range_lp(
                neuron_index
            )
        )

        # ----------------------------------------------------
        # Always active
        # ----------------------------------------------------

        if lower >= 0.0:

            out = star.copy()

            out.activation_pattern.append(
                1
            )

            return [out]


        # ----------------------------------------------------
        # Always inactive
        # ----------------------------------------------------

        if upper <= 0.0:

            out = star.copy()

            out.c[
                neuron_index
            ] = 0.0

            out.V[
                neuron_index,
                :
            ] = 0.0

            out.activation_pattern.append(
                0
            )

            if out.is_empty():
                return []

            return [out]


        # ----------------------------------------------------
        # Crossing zero
        #
        # Active:
        #
        # c_i + V_i alpha >= 0
        #
        # -V_i alpha <= c_i
        # ----------------------------------------------------

        active = star.copy()

        active.C = np.vstack([
            active.C,
            -active.V[
                neuron_index,
                :
            ]
        ])

        active.d = np.concatenate([
            active.d,
            [
                active.c[
                    neuron_index
                ]
            ]
        ])

        active.activation_pattern.append(
            1
        )


        # ----------------------------------------------------
        # Inactive:
        #
        # c_i + V_i alpha <= 0
        #
        # V_i alpha <= -c_i
        # ----------------------------------------------------

        inactive = star.copy()

        inactive.C = np.vstack([
            inactive.C,
            inactive.V[
                neuron_index,
                :
            ]
        ])

        inactive.d = np.concatenate([
            inactive.d,
            [
                -inactive.c[
                    neuron_index
                ]
            ]
        ])

        inactive.c[
            neuron_index
        ] = 0.0

        inactive.V[
            neuron_index,
            :
        ] = 0.0

        inactive.activation_pattern.append(
            0
        )

        output = []

        if not active.is_empty():

            output.append(
                active
            )

        if not inactive.is_empty():

            output.append(
                inactive
            )

        return output


    @staticmethod
    def relu_exact(
        stars
    ):

        if not stars:
            return []

        current = stars

        n_neurons = (
            stars[0].dim
        )

        for neuron_index in range(
            n_neurons
        ):

            next_stars = []

            for star in current:

                next_stars.extend(
                    ExactStar.step_relu(
                        star,
                        neuron_index
                    )
                )

            current = next_stars

        return current


# ============================================================
# Initial Property-2 Star
# ============================================================

def create_input_star():

    C = np.vstack([
        np.eye(5),
        -np.eye(5)
    ])

    d = np.concatenate([
        ALPHA_UB,
        -ALPHA_LB
    ])

    return ExactStar(
        c=PROPERTY2_C.copy(),
        V=PROPERTY2_V.copy(),
        C=C,
        d=d,
        lb=ALPHA_LB.copy(),
        ub=ALPHA_UB.copy()
    )


# ============================================================
# Complete exact propagation
# ============================================================

def exact_propagation(
    network
):

    stars = [
        create_input_star()
    ]

    layer_stats = []

    total_start = (
        time.perf_counter()
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "FULL EXACT STAR PROPAGATION"
    )

    print(
        "=" * 100
    )

    n_affine_layers = len(
        network["W"]
    )

    for layer in range(
        n_affine_layers
    ):

        layer_start = (
            time.perf_counter()
        )

        branches_before = len(
            stars
        )

        # ----------------------------------------------------
        # Affine mapping
        # ----------------------------------------------------

        affine_start = (
            time.perf_counter()
        )

        stars = [
            star.affine_map(
                network["W"][layer],
                network["b"][layer]
            )
            for star in stars
        ]

        affine_seconds = (
            time.perf_counter()
            - affine_start
        )


        # ----------------------------------------------------
        # ReLU for hidden layers only.
        #
        # The final layer is output affine only.
        # ----------------------------------------------------

        relu_seconds = 0.0

        is_output_layer = (
            layer
            ==
            n_affine_layers - 1
        )

        if not is_output_layer:

            relu_start = (
                time.perf_counter()
            )

            stars = (
                ExactStar.relu_exact(
                    stars
                )
            )

            relu_seconds = (
                time.perf_counter()
                - relu_start
            )


        layer_seconds = (
            time.perf_counter()
            - layer_start
        )

        rss_mb = (
            get_process_rss_mb()
        )

        max_rss_mb = (
            get_max_rss_mb()
        )

        row = {
            "layer":
                layer + 1,

            "type":
                (
                    "output_affine"
                    if is_output_layer
                    else "hidden_affine_relu"
                ),

            "branches_before":
                branches_before,

            "branches_after":
                len(stars),

            "affine_seconds":
                affine_seconds,

            "relu_seconds":
                relu_seconds,

            "layer_seconds":
                layer_seconds,

            "rss_mb":
                rss_mb,

            "max_rss_mb":
                max_rss_mb
        }

        layer_stats.append(
            row
        )

        print(
            f"Layer {layer + 1:>2d}/{n_affine_layers}: "
            f"{row['type']:<20s} | "
            f"{branches_before:>8,d} -> "
            f"{len(stars):>8,d} Stars | "
            f"affine={affine_seconds:9.3f}s | "
            f"relu={relu_seconds:9.3f}s | "
            f"total={layer_seconds:9.3f}s | "
            f"RSS={rss_mb:8.1f} MB | "
            f"peakRSS={max_rss_mb:8.1f} MB"
        )

        # Save intermediate layer statistics so that
        # information survives if a later layer is interrupted.
        save_layer_statistics(
            layer_stats,
            LAYER_CSV
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    return (
        stars,
        layer_stats,
        total_seconds
    )


# ============================================================
# Layer-statistics CSV
# ============================================================

def save_layer_statistics(
    rows,
    filename
):

    fields = [
        "layer",
        "type",
        "branches_before",
        "branches_after",
        "affine_seconds",
        "relu_seconds",
        "layer_seconds",
        "rss_mb",
        "max_rss_mb"
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Serialization helpers
# ============================================================

def array_to_json(
    array
):

    return json.dumps(
        np.asarray(
            array
        ).tolist(),
        separators=(",", ":")
    )


def json_to_array(
    text
):

    return np.asarray(
        json.loads(
            text
        ),
        dtype=float
    )


# ============================================================
# PKL
# ============================================================

def star_to_record(
    star
):

    return {
        "c":
            star.c,

        "V":
            star.V,

        "C":
            star.C,

        "d":
            star.d,

        "lb":
            star.lb,

        "ub":
            star.ub,

        "activation_pattern":
            star.activation_pattern
    }


def record_to_star(
    record
):

    return ExactStar(
        c=record["c"],
        V=record["V"],
        C=record["C"],
        d=record["d"],
        lb=record["lb"],
        ub=record["ub"],
        activation_pattern=record[
            "activation_pattern"
        ]
    )


def save_stars_pkl(
    stars,
    filename,
    network,
    layer_stats,
    propagation_seconds
):

    print(
        f"\nSaving {len(stars):,} Stars "
        f"to PKL..."
    )

    start = (
        time.perf_counter()
    )

    payload = {
        "format_version":
            1,

        "description":
            (
                "Distribution-independent exact "
                "ACAS Xu reachable Star geometry"
            ),

        "network_architecture":
            np.asarray(
                network["layer_sizes"]
            ),

        "property2_raw_lb":
            PROPERTY2_RAW_LB,

        "property2_raw_ub":
            PROPERTY2_RAW_UB,

        "property2_x_lb":
            PROPERTY2_X_LB,

        "property2_x_ub":
            PROPERTY2_X_UB,

        "initial_c":
            PROPERTY2_C,

        "initial_V":
            PROPERTY2_V,

        "alpha_lb":
            ALPHA_LB,

        "alpha_ub":
            ALPHA_UB,

        "propagation_seconds":
            propagation_seconds,

        "layer_stats":
            layer_stats,

        "num_final_stars":
            len(stars),

        "stars":
            [
                star_to_record(star)
                for star in stars
            ]
    }

    with open(
        filename,
        "wb"
    ) as f:

        pickle.dump(
            payload,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    seconds = (
        time.perf_counter()
        - start
    )

    size_mb = (
        os.path.getsize(filename)
        / (1024 ** 2)
    )

    print(
        f"PKL save completed: "
        f"{seconds:.2f} s, "
        f"{size_mb:.2f} MB"
    )

    return (
        seconds,
        size_mb
    )


def load_stars_pkl(
    filename
):

    print(
        f"\nLoading PKL: {filename}"
    )

    start = (
        time.perf_counter()
    )

    with open(
        filename,
        "rb"
    ) as f:

        payload = pickle.load(
            f
        )

    stars = [
        record_to_star(
            record
        )
        for record in payload[
            "stars"
        ]
    ]

    seconds = (
        time.perf_counter()
        - start
    )

    print(
        f"Loaded {len(stars):,} Stars "
        f"in {seconds:.2f} s."
    )

    return (
        payload,
        stars,
        seconds
    )


# ============================================================
# CSV
# ============================================================

def save_stars_csv(
    stars,
    filename
):

    print(
        f"\nSaving {len(stars):,} Stars "
        f"to CSV..."
    )

    start = (
        time.perf_counter()
    )

    fields = [
        "branch_id",
        "state_dim",
        "predicate_dim",
        "num_constraints",
        "c",
        "V",
        "C",
        "d",
        "lb",
        "ub",
        "activation_pattern"
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for branch_id, star in enumerate(
            stars
        ):

            writer.writerow({
                "branch_id":
                    branch_id,

                "state_dim":
                    star.dim,

                "predicate_dim":
                    star.n_pred,

                "num_constraints":
                    star.C.shape[0],

                "c":
                    array_to_json(
                        star.c
                    ),

                "V":
                    array_to_json(
                        star.V
                    ),

                "C":
                    array_to_json(
                        star.C
                    ),

                "d":
                    array_to_json(
                        star.d
                    ),

                "lb":
                    array_to_json(
                        star.lb
                    ),

                "ub":
                    array_to_json(
                        star.ub
                    ),

                "activation_pattern":
                    json.dumps(
                        star.activation_pattern,
                        separators=(",", ":")
                    )
            })

            if (
                (branch_id + 1)
                % SAVE_PROGRESS_INTERVAL
                == 0
            ):

                print(
                    f"  CSV saved "
                    f"{branch_id + 1:,}/"
                    f"{len(stars):,}"
                )

    seconds = (
        time.perf_counter()
        - start
    )

    size_mb = (
        os.path.getsize(filename)
        / (1024 ** 2)
    )

    print(
        f"CSV save completed: "
        f"{seconds:.2f} s, "
        f"{size_mb:.2f} MB"
    )

    return (
        seconds,
        size_mb
    )


def load_stars_csv(
    filename
):

    print(
        f"\nLoading CSV: {filename}"
    )

    start = (
        time.perf_counter()
    )

    stars = []

    with open(
        filename,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            stars.append(
                ExactStar(
                    c=json_to_array(
                        row["c"]
                    ),

                    V=json_to_array(
                        row["V"]
                    ),

                    C=json_to_array(
                        row["C"]
                    ),

                    d=json_to_array(
                        row["d"]
                    ),

                    lb=json_to_array(
                        row["lb"]
                    ),

                    ub=json_to_array(
                        row["ub"]
                    ),

                    activation_pattern=
                        json.loads(
                            row[
                                "activation_pattern"
                            ]
                        )
                )
            )

            if (
                len(stars)
                % SAVE_PROGRESS_INTERVAL
                == 0
            ):

                print(
                    f"  CSV loaded "
                    f"{len(stars):,} Stars"
                )

    seconds = (
        time.perf_counter()
        - start
    )

    print(
        f"Loaded {len(stars):,} Stars "
        f"in {seconds:.2f} s."
    )

    return (
        stars,
        seconds
    )


# ============================================================
# Geometry verification
# ============================================================

def verify_star_lists(
    original,
    loaded,
    label
):

    print(
        f"\nVerifying {label}..."
    )

    start = (
        time.perf_counter()
    )

    if len(original) != len(loaded):

        print(
            "FAILED: number of Stars differs."
        )

        return (
            False,
            float("inf"),
            time.perf_counter() - start
        )

    max_error = 0.0

    for i, (a, b) in enumerate(
        zip(
            original,
            loaded
        )
    ):

        arrays_a = [
            a.c,
            a.V,
            a.C,
            a.d,
            a.lb,
            a.ub
        ]

        arrays_b = [
            b.c,
            b.V,
            b.C,
            b.d,
            b.lb,
            b.ub
        ]

        for x, y in zip(
            arrays_a,
            arrays_b
        ):

            if x.shape != y.shape:

                print(
                    f"FAILED at Star {i}: "
                    f"shape mismatch "
                    f"{x.shape} vs {y.shape}"
                )

                return (
                    False,
                    float("inf"),
                    time.perf_counter()
                    - start
                )

            if x.size:

                err = float(
                    np.max(
                        np.abs(
                            x - y
                        )
                    )
                )

                max_error = max(
                    max_error,
                    err
                )

        if (
            a.activation_pattern
            !=
            b.activation_pattern
        ):

            print(
                f"FAILED at Star {i}: "
                "activation pattern mismatch."
            )

            return (
                False,
                float("inf"),
                time.perf_counter()
                - start
            )

    seconds = (
        time.perf_counter()
        - start
    )

    print(
        f"SUCCESS: {label}"
    )

    print(
        f"Maximum numerical difference = "
        f"{max_error:.3e}"
    )

    print(
        f"Verification time = "
        f"{seconds:.2f} s"
    )

    return (
        True,
        max_error,
        seconds
    )


# ============================================================
# Summary
# ============================================================

def save_summary(
    network,
    stars,
    layer_stats,
    propagation_seconds,
    python_peak_mb,
    process_peak_mb,
    pkl_seconds,
    pkl_size_mb,
    pkl_ok,
    csv_seconds,
    csv_size_mb,
    csv_ok
):

    with open(
        SUMMARY_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ACAS Xu Stage-1 Full "
            "Exact-Star Reachability\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        f.write(
            "Network architecture:\n"
        )

        f.write(
            str(
                network[
                    "layer_sizes"
                ]
            )
            + "\n\n"
        )

        f.write(
            "Input property: ACAS Xu Property 2\n"
        )

        f.write(
            "Predicate domain: "
            "alpha in [-1,1]^5\n"
        )

        f.write(
            "Probability distribution: NONE\n\n"
        )

        f.write(
            f"Final Stars: "
            f"{len(stars)}\n"
        )

        f.write(
            f"Exact propagation runtime: "
            f"{propagation_seconds:.6f} s\n"
        )

        f.write(
            f"Python tracemalloc peak: "
            f"{python_peak_mb:.3f} MB\n"
        )

        f.write(
            f"Process max RSS: "
            f"{process_peak_mb:.3f} MB\n\n"
        )

        f.write(
            "PKL output:\n"
        )

        f.write(
            f"  file = {PKL_FILE}\n"
        )

        f.write(
            f"  save time = "
            f"{pkl_seconds:.6f} s\n"
        )

        f.write(
            f"  size = "
            f"{pkl_size_mb:.3f} MB\n"
        )

        f.write(
            f"  round-trip success = "
            f"{pkl_ok}\n\n"
        )

        if SAVE_CSV:

            f.write(
                "CSV output:\n"
            )

            f.write(
                f"  file = {CSV_FILE}\n"
            )

            f.write(
                f"  save time = "
                f"{csv_seconds:.6f} s\n"
            )

            f.write(
                f"  size = "
                f"{csv_size_mb:.3f} MB\n"
            )

            f.write(
                f"  round-trip success = "
                f"{csv_ok}\n\n"
            )

        f.write(
            "Layer statistics:\n"
        )

        for row in layer_stats:

            f.write(
                f"Layer {row['layer']}: "
                f"type={row['type']}, "
                f"branches="
                f"{row['branches_before']} -> "
                f"{row['branches_after']}, "
                f"affine="
                f"{row['affine_seconds']:.6f}s, "
                f"relu="
                f"{row['relu_seconds']:.6f}s, "
                f"total="
                f"{row['layer_seconds']:.6f}s, "
                f"rss="
                f"{row['rss_mb']:.3f}MB, "
                f"max_rss="
                f"{row['max_rss_mb']:.3f}MB\n"
            )


# ============================================================
# Main
# ============================================================

def run_one_network(network_id, mat_path, output_dir):

    global PKL_FILE, CSV_FILE, LAYER_CSV, SUMMARY_TXT

    output_dir.mkdir(parents=True, exist_ok=True)
    PKL_FILE = str(output_dir / "acasxu_reachable_stars.pkl")
    CSV_FILE = str(output_dir / "acasxu_reachable_stars.csv")
    LAYER_CSV = str(output_dir / "acasxu_stage1_layers.csv")
    SUMMARY_TXT = str(output_dir / "acasxu_stage1_summary.txt")

    print(
        "=" * 100
    )

    print(
        "ACAS Xu -- STAGE 1 "
        "FULL DISTRIBUTION-INDEPENDENT "
        "EXACT-STAR REACHABILITY"
    )

    print(
        "=" * 100
    )


    print(f"\nBatch network ID = {network_id}")
    print(f"Network file = {mat_path}")
    print(f"Output folder = {output_dir}")

    # --------------------------------------------------------
    # Load network
    # --------------------------------------------------------

    network = load_acasxu_mat(
        mat_path
    )

    print(
        "\nNetwork architecture:"
    )

    print(
        network[
            "layer_sizes"
        ]
    )

    print(
        f"Affine layers = "
        f"{len(network['W'])}"
    )

    print(
        f"Hidden ReLU layers = "
        f"{len(network['W']) - 1}"
    )


    # --------------------------------------------------------
    # Initial Star
    # --------------------------------------------------------

    print(
        "\nProperty-2 normalized input:"
    )

    print(
        "x_lb =",
        PROPERTY2_X_LB
    )

    print(
        "x_ub =",
        PROPERTY2_X_UB
    )

    print(
        "\nInitial Star:"
    )

    print(
        "c =",
        PROPERTY2_C
    )

    print(
        "diag(V) =",
        PROPERTY2_V_DIAG
    )

    print(
        "alpha in [-1,1]^5"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No probability distribution is "
        "used during propagation."
    )


    # --------------------------------------------------------
    # Memory instrumentation
    # --------------------------------------------------------

    tracemalloc.start()


    # --------------------------------------------------------
    # Full propagation
    # --------------------------------------------------------

    (
        stars,
        layer_stats,
        propagation_seconds
    ) = exact_propagation(
        network
    )


    _, python_peak_bytes = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    python_peak_mb = (
        python_peak_bytes
        / (1024 ** 2)
    )

    process_peak_mb = (
        get_max_rss_mb()
    )


    print(
        "\n" + "=" * 100
    )

    print(
        "PROPAGATION COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"Final Stars = "
        f"{len(stars):,}"
    )

    print(
        f"Exact propagation time = "
        f"{propagation_seconds:.2f} s"
    )

    print(
        f"Python peak memory = "
        f"{python_peak_mb:.2f} MB"
    )

    print(
        f"Process peak RSS = "
        f"{process_peak_mb:.2f} MB"
    )


    # --------------------------------------------------------
    # Save PKL
    # --------------------------------------------------------

    (
        pkl_seconds,
        pkl_size_mb
    ) = save_stars_pkl(
        stars,
        PKL_FILE,
        network,
        layer_stats,
        propagation_seconds
    )


    # --------------------------------------------------------
    # Verify PKL
    # --------------------------------------------------------

    pkl_ok = None

    if VERIFY_PKL:

        (
            pkl_payload,
            pkl_stars,
            pkl_load_seconds
        ) = load_stars_pkl(
            PKL_FILE
        )

        (
            pkl_ok,
            pkl_error,
            pkl_verify_seconds
        ) = verify_star_lists(
            stars,
            pkl_stars,
            "PKL round trip"
        )

        # Free duplicate copy before CSV.
        del pkl_stars
        del pkl_payload


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_seconds = float("nan")
    csv_size_mb = float("nan")
    csv_ok = None

    if SAVE_CSV:

        (
            csv_seconds,
            csv_size_mb
        ) = save_stars_csv(
            stars,
            CSV_FILE
        )


        # ----------------------------------------------------
        # Verify CSV
        # ----------------------------------------------------

        if VERIFY_CSV:

            (
                csv_stars,
                csv_load_seconds
            ) = load_stars_csv(
                CSV_FILE
            )

            (
                csv_ok,
                csv_error,
                csv_verify_seconds
            ) = verify_star_lists(
                stars,
                csv_stars,
                "CSV round trip"
            )

            del csv_stars


    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_summary(
        network=network,
        stars=stars,
        layer_stats=layer_stats,
        propagation_seconds=
            propagation_seconds,
        python_peak_mb=
            python_peak_mb,
        process_peak_mb=
            process_peak_mb,
        pkl_seconds=
            pkl_seconds,
        pkl_size_mb=
            pkl_size_mb,
        pkl_ok=
            pkl_ok,
        csv_seconds=
            csv_seconds,
        csv_size_mb=
            csv_size_mb,
        csv_ok=
            csv_ok
    )


    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "FINAL STAGE-1 RESULT"
    )

    print(
        "=" * 100
    )

    print(
        f"Final Stars: "
        f"{len(stars):,}"
    )

    print(
        f"Propagation time: "
        f"{propagation_seconds:.2f} s"
    )

    print(
        f"Python peak memory: "
        f"{python_peak_mb:.2f} MB"
    )

    print(
        f"Process peak RSS: "
        f"{process_peak_mb:.2f} MB"
    )

    print(
        f"\nPKL:"
    )

    print(
        f"  {PKL_FILE}"
    )

    print(
        f"  size = "
        f"{pkl_size_mb:.2f} MB"
    )

    print(
        f"  round trip = "
        f"{pkl_ok}"
    )

    if SAVE_CSV:

        print(
            f"\nCSV:"
        )

        print(
            f"  {CSV_FILE}"
        )

        print(
            f"  size = "
            f"{csv_size_mb:.2f} MB"
        )

        print(
            f"  round trip = "
            f"{csv_ok}"
        )

    print(
        f"\nLayer statistics:"
    )

    print(
        f"  {LAYER_CSV}"
    )

    print(
        f"\nSummary:"
    )

    print(
        f"  {SUMMARY_TXT}"
    )

    print(
        "=" * 100
    )


def network_id_to_indices(network_id):
    if not 1 <= network_id <= 45:
        raise ValueError("network_id must be between 1 and 45.")
    i = (network_id - 1) // 9 + 1
    j = (network_id - 1) % 9 + 1
    return i, j


def find_network_mat(i, j):
    candidate = Path("ACASXU") / (
        f"ACASXU_run2a_{i}_{j}_batch_2000.mat"
    )

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Cannot find ACAS Xu network ({i},{j}). "
        f"Expected: {candidate}"
    )


def main():
    if not TARGET_NETWORKS:
        raise ValueError("TARGET_NETWORKS must not be empty.")
    if any(not 1 <= network_id <= 45 for network_id in TARGET_NETWORKS):
        raise ValueError("Every target network ID must be between 1 and 45.")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("ACAS Xu -- SELECTED PROPERTY-2 EXACT-STAR REACHABILITY")
    print(f"Selected flat IDs: {TARGET_NETWORKS}")
    print("=" * 100)

    for run_index, network_id in enumerate(TARGET_NETWORKS, start=1):
        i, j = network_id_to_indices(network_id)
        mat_path = find_network_mat(i, j)
        output_dir = RESULTS_ROOT / str(network_id)

        print("\n" + "#" * 100)
        print(
            f"START {run_index}/{len(TARGET_NETWORKS)} -> "
            f"flat ID {network_id} -> ACAS Xu ({i},{j})"
        )
        print("#" * 100)

        run_one_network(network_id, mat_path, output_dir)

        print("#" * 100)
        print(
            f"COMPLETED {run_index}/{len(TARGET_NETWORKS)} -> "
            f"flat ID {network_id} -> {RESULTS_ROOT}/{network_id}/"
        )
        print("#" * 100)


if __name__ == "__main__":
    main()
