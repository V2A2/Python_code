#!/usr/bin/env python3
"""Property 3: run exact propagation and unsafe probability computation in one pass.

Requires P3_propagation.py, P3_unsafe_compute.py and the ACASXU/ network files.
Unlike the two-stage workflow, no reachable-Star PKL/CSV is written.
"""

import argparse
import gc
from pathlib import Path

import P3_propagation as propagation
import P3_unsafe_compute as probability


def run(start_network=1, end_network=45):
    if not (1 <= start_network <= end_network <= 45):
        raise ValueError("Require 1 <= start_network <= end_network <= 45.")

    output_root = Path("P3_result")
    output_root.mkdir(parents=True, exist_ok=True)
    master_csv = output_root / "acasxu_all_probability_results.csv"

    # Identical quadrature and fixed densities to P3_unsafe_compute.py.
    points, weights = probability.build_quadrature()
    density_true = probability.true_density(points)
    density_ps = probability.probstar_density(points)
    density_gps = probability.gps_density(points)

    rows = []
    for network_id in range(start_network, end_network + 1):
        i, j = propagation.network_id_to_indices(network_id)
        mat_path = propagation.find_network_mat(i, j)
        print(f"\n{'#' * 80}\nNETWORK {network_id}/45 -- ACAS Xu ({i},{j})\n{'#' * 80}")

        network = propagation.load_acasxu_mat(mat_path)
        # Keep the original layer-statistics output inside this network's folder.
        network_dir = output_root / str(network_id)
        network_dir.mkdir(parents=True, exist_ok=True)
        propagation.LAYER_CSV = str(network_dir / "acasxu_stage1_layers.csv")
        stars, layer_stats, propagation_seconds = propagation.exact_propagation(network)
        num_final_stars = len(stars)

        # The second-stage code expects dictionaries with c, V, C, d, lb, ub.
        # Convert in memory rather than writing and reloading a large PKL.
        records = [propagation.star_to_record(star) for star in stars]
        del stars
        unsafe, geometry_time = probability.prepare_unsafe_polytopes(records)
        num_unsafe_stars = len(unsafe)
        del records

        if num_unsafe_stars:
            result_true = probability.compute_probability(
                unsafe, points, weights, density_true, "TRUE STUDENT-t MIXTURE")
            result_ps = probability.compute_probability(
                unsafe, points, weights, density_ps, "ORIGINAL PROBSTAR")
            result_gps = probability.compute_probability(
                unsafe, points, weights, density_gps, "GENERALIZED PROBSTAR K=10")
        else:
            zero = {"probability": 0.0, "time": 0.0, "peak_memory_mb": 0.0}
            result_true = result_ps = result_gps = zero

        p_true = result_true["probability"]
        p_ps = result_ps["probability"]
        p_gps = result_gps["probability"]
        err_ps = abs(p_ps - p_true)
        err_gps = abs(p_gps - p_true)
        row = {
            "network_id": network_id,
            "num_final_stars": num_final_stars,
            "num_unsafe_stars": num_unsafe_stars,
            "geometry_time_s": geometry_time,
            "p_true": p_true,
            "p_probstar": p_ps,
            "p_gps": p_gps,
            "probstar_abs_error": err_ps,
            "gps_abs_error": err_gps,
            "gps_error_reduction_percent": (1 - err_gps / err_ps) * 100 if err_ps > 0 else 0.0,
            "true_probability_time_s": result_true["time"],
            "true_peak_memory_mb": result_true["peak_memory_mb"],
            "probstar_probability_time_s": result_ps["time"],
            "probstar_peak_memory_mb": result_ps["peak_memory_mb"],
            "gps_probability_time_s": result_gps["time"],
            "gps_peak_memory_mb": result_gps["peak_memory_mb"],
            "gps_to_probstar_time_ratio": (
                result_gps["time"] / result_ps["time"] if result_ps["time"] > 0 else 0.0
            ),
        }
        probability.write_result_csv(network_dir / probability.NETWORK_RESULT_CSV, [row])
        rows.append(row)
        probability.write_result_csv(master_csv, rows)
        print(f"Completed network {network_id}: P_true={p_true:.12e}, "
              f"P_ProbStar={p_ps:.12e}, P_GPS={p_gps:.12e}")
        print(f"Propagation time: {propagation_seconds:.3f} s; "
              f"results: {network_dir / probability.NETWORK_RESULT_CSV}")
        del unsafe, network, layer_stats
        gc.collect()

    print(f"\nCompleted {len(rows)} networks. Master results: {master_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1, help="First network ID (default: 1)")
    parser.add_argument("--end", type=int, default=45, help="Last network ID (default: 45)")
    args = parser.parse_args()
    run(args.start, args.end)
