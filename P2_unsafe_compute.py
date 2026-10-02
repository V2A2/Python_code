#!/usr/bin/env python3
"""
ACAS Xu Property 2, batch networks 11, 19, 24, 25, 28, 34:
local-polytope deterministic probability integration.

Uses the already saved exact reachable Stars. For every P2-unsafe branch:
1) find a tight 5-D axis-aligned bounding box by LP;
2) place a local Gauss-Legendre tensor grid inside that box;
3) retain points satisfying the exact polytope constraints;
4) evaluate TRUE / ProbStar / GPS densities on the same accepted points;
5) sum branch probabilities.

No network propagation, no fitting, no MC/QMC.
"""
import csv, pickle, time, tracemalloc
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.stats import multivariate_normal, multivariate_t
from numpy.polynomial.legendre import leggauss

RESULTS_ROOT = Path("results_property2")
NETWORK_IDS = [11, 19, 24, 25, 28, 34]
PKL_NAME = "acasxu_reachable_stars.pkl"
OUTPUT_NAME = "acasxu_probability_results_local.csv"
DIM = 5
LOCAL_GL_ORDER = 4
TOL = 1e-10

# P2 unsafe: y0 >= y1,y2,y3,y4, written H y <= 0
H = np.array([
    [-1., 1., 0., 0., 0.],
    [-1., 0., 1., 0., 0.],
    [-1., 0., 0., 1., 0.],
    [-1., 0., 0., 0., 1.]
])
g = np.zeros(4)

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




def load_stars(path):
    print(f"Loading {path} ...")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        stars = obj
    elif isinstance(obj, dict):
        for key in ("stars","final_stars","reachable_stars","results"):
            if key in obj and isinstance(obj[key], list):
                stars = obj[key]
                break
        else:
            raise RuntimeError("Cannot locate Star list in pickle.")
    else:
        raise RuntimeError(f"Unsupported pickle type: {type(obj)}")
    print(f"Loaded final Stars: {len(stars):,}")
    return stars

def get_array(star, name):
    if isinstance(star, dict):
        return np.asarray(star[name], dtype=float)
    return np.asarray(getattr(star, name), dtype=float)

def make_unsafe_polytope(star):
    c=get_array(star,"c").reshape(-1)
    V=get_array(star,"V")
    C=get_array(star,"C")
    d=get_array(star,"d").reshape(-1)
    lb=get_array(star,"lb").reshape(-1)
    ub=get_array(star,"ub").reshape(-1)
    if C.size==0: C=np.zeros((0,DIM))
    elif C.ndim==1: C=C.reshape(1,-1)
    Cu=H@V
    du=g-H@c
    return np.vstack((C,Cu)), np.concatenate((d,du)), lb, ub

def feasible(C,d,lb,ub):
    r=linprog(np.zeros(DIM),A_ub=C,b_ub=d,
              bounds=list(zip(lb,ub)),method="highs")
    return r.success

def local_bounds(C,d,lb,ub):
    """Tight coordinate-wise bounding box of the polytope."""
    lo=np.empty(DIM); hi=np.empty(DIM)
    bounds=list(zip(lb,ub))
    for j in range(DIM):
        e=np.zeros(DIM); e[j]=1.0
        rmin=linprog(e,A_ub=C,b_ub=d,bounds=bounds,method="highs")
        rmax=linprog(-e,A_ub=C,b_ub=d,bounds=bounds,method="highs")
        if not (rmin.success and rmax.success):
            return None,None
        lo[j]=rmin.fun
        hi[j]=-rmax.fun
    return lo,hi

nodes_1d, weights_1d = leggauss(LOCAL_GL_ORDER)
base_grids=np.meshgrid(*([nodes_1d]*DIM),indexing="ij")
BASE_POINTS=np.stack([x.ravel() for x in base_grids],axis=1)
wg=np.meshgrid(*([weights_1d]*DIM),indexing="ij")
BASE_WEIGHTS=np.ones(len(BASE_POINTS))
for w in wg: BASE_WEIGHTS*=w.ravel()

def local_quadrature(lo,hi):
    half=0.5*(hi-lo)
    mid=0.5*(hi+lo)
    pts=mid[None,:]+BASE_POINTS*half[None,:]
    # Jacobian for mapping [-1,1]^5 -> local box
    qw=BASE_WEIGHTS*np.prod(half)
    return pts,qw

def run_network(network_id):
    path = RESULTS_ROOT / str(network_id) / PKL_NAME
    out = RESULTS_ROOT / str(network_id) / OUTPUT_NAME

    print("\n" + "=" * 100)
    print(f"ACAS Xu Property 2 -- NETWORK {network_id} LOCAL POLYTOPE INTEGRATION")
    print("=" * 100)
    print("Unsafe region   : y0 >= y1, y0 >= y2, y0 >= y3, y0 >= y4")
    print(f"Local GL order  : {LOCAL_GL_ORDER}")
    print(f"Points/polytope : {LOCAL_GL_ORDER**DIM:,}")
    print("Same accepted local points are used for TRUE / ProbStar / GPS.")

    if not path.exists():
        print(f"ERROR: missing input file: {path}")
        return None

    stars = load_stars(path)
    p_true = p_ps = p_gps = 0.0
    nunsafe = nhit = nboundfail = 0
    accepted_total = 0
    t0 = time.perf_counter()
    tracemalloc.start()

    for si, star in enumerate(stars, 1):
        C, d, lb, ub = make_unsafe_polytope(star)
        if not feasible(C, d, lb, ub):
            continue
        nunsafe += 1

        lo, hi = local_bounds(C, d, lb, ub)
        if lo is None:
            nboundfail += 1
            continue

        # Guard tiny negative widths from LP roundoff.
        widths = hi - lo
        if np.any(widths <= 0):
            continue

        pts, qw = local_quadrature(lo, hi)
        inside = np.all(pts @ C.T <= d[None, :] + TOL, axis=1)

        if np.any(inside):
            nhit += 1
            q = qw[inside]
            x = pts[inside]
            accepted_total += len(x)

            # Evaluate the same three distributions on the accepted local points.
            p_true += float(np.sum(q * true_density(x)))
            p_ps   += float(np.sum(q * probstar_density(x)))
            p_gps  += float(np.sum(q * gps_density(x)))

        if nunsafe % 500 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"Network {network_id:2d} | Unsafe {nunsafe:6,d} | "
                f"local-hit {nhit:6,d} | accepted pts {accepted_total:10,d} | "
                f"Ptrue={p_true:.6e} | PPS={p_ps:.6e} | PGPS={p_gps:.6e} | "
                f"{elapsed:9.1f}s"
            )

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024**2)

    err_ps = abs(p_ps - p_true)
    err_gps = abs(p_gps - p_true)
    reduction = (err_ps - err_gps) / err_ps * 100.0 if err_ps > 0 else 0.0

    print("\n" + "=" * 100)
    print(f"FINAL RESULT -- NETWORK {network_id}")
    print("=" * 100)
    print(f"Final Stars             : {len(stars):,}")
    print(f"Unsafe-intersecting     : {nunsafe:,}")
    print(f"Local-grid-hit          : {nhit:,}")
    print(f"Bounding-box LP failures: {nboundfail:,}")
    print(f"Accepted local points   : {accepted_total:,}")
    print(f"P_true(unsafe)          : {p_true:.12e}")
    print(f"P_ProbStar(unsafe)      : {p_ps:.12e}")
    print(f"P_GPS(unsafe)           : {p_gps:.12e}")
    print(f"ProbStar abs. error     : {err_ps:.12e}")
    print(f"GPS abs. error          : {err_gps:.12e}")
    print(f"GPS error reduction     : {reduction:.3f}%")
    print(f"Total time              : {elapsed:.3f} s")
    print(f"Peak Python memory      : {peak_mb:.3f} MB")
    print("=" * 100)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "network", "local_gl_order", "final_stars", "unsafe_intersecting",
            "local_grid_hit", "accepted_points", "p_true", "p_probstar", "p_gps",
            "probstar_abs_error", "gps_abs_error",
            "gps_error_reduction_percent", "time_seconds", "peak_memory_mb"
        ])
        w.writerow([
            network_id, LOCAL_GL_ORDER, len(stars), nunsafe, nhit, accepted_total,
            p_true, p_ps, p_gps, err_ps, err_gps, reduction, elapsed, peak_mb
        ])
    print(f"Saved: {out}")

    return {
        "network": network_id,
        "local_gl_order": LOCAL_GL_ORDER,
        "final_stars": len(stars),
        "unsafe_intersecting": nunsafe,
        "local_grid_hit": nhit,
        "accepted_points": accepted_total,
        "p_true": p_true,
        "p_probstar": p_ps,
        "p_gps": p_gps,
        "probstar_abs_error": err_ps,
        "gps_abs_error": err_gps,
        "gps_error_reduction_percent": reduction,
        "time_seconds": elapsed,
        "peak_memory_mb": peak_mb,
    }


def main():
    print("=" * 100)
    print("ACAS Xu Property 2 -- BATCH LOCAL POLYTOPE PROBABILITY INTEGRATION")
    print("=" * 100)
    print(f"Networks        : {NETWORK_IDS}")
    print(f"Results root    : {RESULTS_ROOT}")
    print(f"Local GL order  : {LOCAL_GL_ORDER}")
    print(f"Points/polytope : {LOCAL_GL_ORDER**DIM:,}")
    print("All six selected Property-2 networks use the same numerical procedure and output format.")
    
    all_results = []
    batch_t0 = time.perf_counter()

    for network_id in NETWORK_IDS:
        result = run_network(network_id)
        if result is not None:
            all_results.append(result)

    # In addition to the identical per-network CSV files, save one convenient batch summary.
    summary_path = RESULTS_ROOT / "p2_probability_results_local_selected6.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_results)

    batch_elapsed = time.perf_counter() - batch_t0

    print("\n" + "=" * 100)
    print("BATCH COMPLETE")
    print("=" * 100)
    print(f"Completed networks : {[r['network'] for r in all_results]}")
    print(f"Batch total time   : {batch_elapsed:.3f} s")
    if all_results:
        print(f"Combined summary   : {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
