#!/usr/bin/env python3
"""Run ACAS Xu Property 2 propagation and unsafe probability integration in memory.

Requires P2_propagation.py, P2_unsafe_compute.py, and ACASXU/*.mat.
No intermediate reachable-Star PKL/CSV files are written.
Uses the exact same propagation, densities, and quadrature routines as the
original two-stage scripts. Results are saved as small CSV files.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

import P2_propagation as propagation
import P2_unsafe_compute as probability


def compute_from_stars(stars, network_id):
    """Same per-Star integration algorithm as P2_unsafe_compute.run_network."""
    p_true = p_ps = p_gps = 0.0
    nunsafe = nhit = nboundfail = accepted_total = 0
    start = time.perf_counter()

    for star in stars:
        C, d, lb, ub = probability.make_unsafe_polytope(star)
        if not probability.feasible(C, d, lb, ub):
            continue
        nunsafe += 1
        lo, hi = probability.local_bounds(C, d, lb, ub)
        if lo is None:
            nboundfail += 1
            continue
        if np.any(hi - lo <= 0):
            continue
        pts, qw = probability.local_quadrature(lo, hi)
        inside = np.all(pts @ C.T <= d[None, :] + probability.TOL, axis=1)
        if np.any(inside):
            nhit += 1
            x, q = pts[inside], qw[inside]
            accepted_total += len(x)
            p_true += float(np.sum(q * probability.true_density(x)))
            p_ps += float(np.sum(q * probability.probstar_density(x)))
            p_gps += float(np.sum(q * probability.gps_density(x)))
        if nunsafe % 500 == 0:
            print(f"  Network {network_id}: unsafe branches {nunsafe:,}, "
                  f"integrated branches {nhit:,}", flush=True)

    err_ps = abs(p_ps - p_true)
    err_gps = abs(p_gps - p_true)
    return {
        "network": network_id,
        "local_gl_order": probability.LOCAL_GL_ORDER,
        "final_stars": len(stars),
        "unsafe_intersecting": nunsafe,
        "local_grid_hit": nhit,
        "accepted_points": accepted_total,
        "p_true": p_true,
        "p_probstar": p_ps,
        "p_gps": p_gps,
        "probstar_abs_error": err_ps,
        "gps_abs_error": err_gps,
        "gps_error_reduction_percent": (
            (err_ps - err_gps) / err_ps * 100.0 if err_ps > 0 else 0.0
        ),
        "time_seconds": time.perf_counter() - start,
        "bounding_box_lp_failures": nboundfail,
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--networks", nargs="+", type=int,
                        default=[11, 19, 24, 25, 28, 34])
    parser.add_argument("--output", type=Path,
                        default=Path("results_property2"))
    args = parser.parse_args()
    rows = []
    for network_id in args.networks:
        i, j = propagation.network_id_to_indices(network_id)
        mat_file = propagation.find_network_mat(i, j)
        print(f"\n=== Network {network_id} ({i},{j}): exact propagation ===", flush=True)
        network = propagation.load_acasxu_mat(mat_file)
        stars, layer_stats, propagation_seconds = propagation.exact_propagation(network)
        print(f"\n=== Network {network_id}: unsafe probability integration ===", flush=True)
        result = compute_from_stars(stars, network_id)
        result["propagation_seconds"] = propagation_seconds
        result["total_seconds"] = propagation_seconds + result["time_seconds"]
        rows.append(result)
        write_csv(args.output / str(network_id) /
                  probability.OUTPUT_NAME, [result])
        print(f"Network {network_id}: P_true={result['p_true']:.12e}, "
              f"P_PS={result['p_probstar']:.12e}, "
              f"P_GPS={result['p_gps']:.12e}", flush=True)
        del stars
    write_csv(args.output / "p2_probability_results_local_selected6.csv", rows)


if __name__ == "__main__":
    main()
